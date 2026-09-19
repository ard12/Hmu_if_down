/**
 * @file main.c
 * @brief Tracker Nodes (Nodes 1, 2, 3): Wi-Fi CSI Receiver and Streamer
 * 
 * Configures CSI collection callback on ESP32/ESP32-C6, extracts raw
 * subcarrier I/Q information, packages with Node ID, and streams via UDP
 * to the central Fall Detection processing hub.
 *
 * IMPORTANT: The CSI callback runs in the Wi-Fi driver's high-priority task.
 * All network I/O is decoupled via a FreeRTOS Queue to prevent WDT resets.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "mdns.h"

#if CONFIG_OTA_ENABLED
#include "esp_ota_ops.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#ifndef CONFIG_OTA_VERSION
#define CONFIG_OTA_VERSION "3.1.0"
#endif
#ifndef CONFIG_OTA_PORT
#define CONFIG_OTA_PORT 8000
#endif
#endif

static const char *TAG = "CSI_TRACKER";

// Configure Node ID: 1, 2, or 3 (change per flashed board)
#ifndef CONFIG_TRACKER_NODE_ID
#define CONFIG_TRACKER_NODE_ID 1
#endif

#ifndef CONFIG_ROOM_ID
#define CONFIG_ROOM_ID 1
#endif

#define WIFI_CHANNEL 6
#define DEST_PORT 5555
// Fallback subnet broadcast IP if mDNS discovery is unavailable
#define DEST_IP "255.255.255.255"
#define MDNS_HUB_HOST "falldetect-hub"

#define CSI_QUEUE_DEPTH 16
#define MAX_SUBCARRIERS 128
#define HEARTBEAT_INTERVAL_MS 5000

// Header for UDP CSI Stream Packet
typedef struct __attribute__((packed)) {
    uint8_t magic[4];          // "CSIF"
    uint8_t node_id;           // 1, 2, or 3
    uint8_t room_id;           // compile-time room identifier (default 0x01)
    int8_t rssi;               // Packet RSSI in dBm
    uint8_t _pad;              // alignment pad (was unused byte)
    uint16_t subcarrier_count; // Number of subcarriers
    uint32_t timestamp_ms;
    uint32_t seq_num;
} csi_udp_header_t;

// Queue item: pre-built UDP packet ready to send
typedef struct {
    uint16_t total_len;
    uint8_t buffer[sizeof(csi_udp_header_t) + MAX_SUBCARRIERS * 2];
} csi_queue_item_t;

static int s_udp_sock = -1;
static struct sockaddr_in s_dest_addr;
static uint32_t s_packet_counter = 0;
static QueueHandle_t s_csi_queue = NULL;

// Counters for diagnostics
static volatile uint32_t s_queue_drops = 0;
static volatile uint32_t s_packets_sent = 0;

// Optional AP MAC filtering: set to 1 and define target MAC to reject crosstalk/ambient Wi-Fi
#ifndef CONFIG_FILTER_AP_MAC
#define CONFIG_FILTER_AP_MAC 0
#endif

#if CONFIG_FILTER_AP_MAC
// Replace with Transmitter AP MAC printed on Node 0 startup log
static const uint8_t s_target_ap_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
#endif

static void wifi_csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len == 0 || s_csi_queue == NULL) {
        return;
    }

#if CONFIG_FILTER_AP_MAC
    if (memcmp(info->mac, s_target_ap_mac, 6) != 0) {
        return; // Discard packets from non-AP sources
    }
#endif

    uint16_t subcarrier_count = info->len / 2;
    if (subcarrier_count > MAX_SUBCARRIERS) {
        subcarrier_count = MAX_SUBCARRIERS;
    }

    // Build packet in a stack-local queue item (fast, no heap alloc)
    csi_queue_item_t item;
    csi_udp_header_t *hdr = (csi_udp_header_t *)item.buffer;

    memcpy(hdr->magic, "CSIF", 4);
    hdr->node_id = (uint8_t)CONFIG_TRACKER_NODE_ID;
    hdr->room_id = (uint8_t)CONFIG_ROOM_ID;
    hdr->rssi = info->rx_ctrl.rssi;
    hdr->_pad = 0;
    hdr->subcarrier_count = subcarrier_count;
    hdr->timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);
    hdr->seq_num = s_packet_counter++;

    // Copy raw I/Q subcarrier bytes
    memcpy(item.buffer + sizeof(csi_udp_header_t), info->buf, subcarrier_count * 2);
    item.total_len = sizeof(csi_udp_header_t) + (subcarrier_count * 2);

    // Non-blocking enqueue — drop packet if queue is full rather than blocking
    if (xQueueSendFromISR(s_csi_queue, &item, NULL) != pdTRUE) {
        s_queue_drops++;
    }
}

/**
 * @brief Dedicated UDP transmission task.
 * Reads pre-built packets from the queue and sends them over the network.
 * This runs in its own FreeRTOS task context where blocking I/O is safe.
 */
static void udp_tx_task(void *arg) {
    csi_queue_item_t item;
    while (1) {
        if (xQueueReceive(s_csi_queue, &item, portMAX_DELAY) == pdTRUE) {
            if (s_udp_sock >= 0) {
                sendto(s_udp_sock, item.buffer, item.total_len, 0,
                       (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
                s_packets_sent++;
            }
        }
    }
}

/**
 * @brief Periodic heartbeat task for diagnostics and graceful degradation.
 * Logs queue health and sends a heartbeat beacon to the hub.
 */
static void heartbeat_task(void *arg) {
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(HEARTBEAT_INTERVAL_MS));

        ESP_LOGI(TAG, "[Node %d] Sent: %lu | QueueDrops: %lu | QueueFree: %d/%d",
                 CONFIG_TRACKER_NODE_ID,
                 (unsigned long)s_packets_sent,
                 (unsigned long)s_queue_drops,
                 (int)uxQueueSpacesAvailable(s_csi_queue),
                 CSI_QUEUE_DEPTH);

        // Send heartbeat JSON to hub so it knows this node is alive
        if (s_udp_sock >= 0) {
            char hb[96];
            int len = snprintf(hb, sizeof(hb),
                               "{\"heartbeat\":%d,\"sent\":%lu,\"drops\":%lu}\n",
                               CONFIG_TRACKER_NODE_ID,
                               (unsigned long)s_packets_sent,
                               (unsigned long)s_queue_drops);
            sendto(s_udp_sock, hb, len, 0,
                   (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
        }
    }
}

static void resolve_hub_mdns(void) {
    esp_err_t err = mdns_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "mDNS init failed (%s), defaulting to broadcast %s", esp_err_to_name(err), DEST_IP);
        return;
    }

    ESP_LOGI(TAG, "Resolving hub IP via mDNS query for '%s.local'...", MDNS_HUB_HOST);
    esp_ip4_addr_t addr;
    addr.addr = 0;

    err = mdns_query_a(MDNS_HUB_HOST, 2500, &addr);
    if (err == ESP_OK && addr.addr != 0) {
        s_dest_addr.sin_addr.s_addr = addr.addr;
        ESP_LOGI(TAG, "mDNS successfully resolved %s to " IPSTR " (unicast mode)", MDNS_HUB_HOST, IP2STR(&addr));
    } else {
        ESP_LOGW(TAG, "mDNS query timed out; falling back to subnet broadcast: %s", DEST_IP);
    }
}

static void init_udp_socket(void) {
    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "Unable to create UDP socket: errno %d", errno);
        return;
    }

    int broadcast_enable = 1;
    setsockopt(s_udp_sock, SOL_SOCKET, SO_BROADCAST, &broadcast_enable, sizeof(broadcast_enable));

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(DEST_PORT);
    s_dest_addr.sin_addr.s_addr = inet_addr(DEST_IP);

    // Attempt mDNS zero-configuration resolution of hub hostname
    resolve_hub_mdns();

    ESP_LOGI(TAG, "UDP socket ready, destination port: %d", DEST_PORT);
}

static void init_wifi_csi(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Lock to transmitter channel
    ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));

    // Enable CSI capture
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI collection enabled for Tracker Node %d on Channel %d",
             CONFIG_TRACKER_NODE_ID, WIFI_CHANNEL);
}

#if CONFIG_OTA_ENABLED
static void check_and_apply_ota(void) {
    char ota_url[128];
    snprintf(ota_url, sizeof(ota_url),
             "http://" IPSTR ":%d/firmware/tracker_v" CONFIG_OTA_VERSION ".bin",
             IP2STR(&s_dest_addr.sin_addr), CONFIG_OTA_PORT);
    ESP_LOGI(TAG, "[OTA] Checking for update: %s", ota_url);

    esp_http_client_config_t http_cfg = {
        .url = ota_url,
        .timeout_ms = 8000,
        .keep_alive_enable = false,
    };
    esp_https_ota_config_t ota_cfg = { .http_config = &http_cfg };
    esp_err_t err = esp_https_ota(&ota_cfg);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "[OTA] Update succeeded — rebooting to new firmware.");
        esp_restart();
    } else {
        ESP_LOGW(TAG, "[OTA] No update applied (err=%s) — continuing normal boot.", esp_err_to_name(err));
    }
}
#endif

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Create the CSI packet queue BEFORE enabling CSI
    s_csi_queue = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(csi_queue_item_t));
    if (s_csi_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create CSI queue!");
        return;
    }

    init_udp_socket();
#if CONFIG_OTA_ENABLED
    check_and_apply_ota();
#endif
    init_wifi_csi();

    // Launch the dedicated UDP transmission task (safe context for sendto)
    xTaskCreate(udp_tx_task, "udp_tx_task", 4096, NULL, 5, NULL);

    // Launch heartbeat/diagnostics task
    xTaskCreate(heartbeat_task, "heartbeat_task", 2048, NULL, 3, NULL);

    ESP_LOGI(TAG, "Tracker Node %d is actively running and streaming CSI.", CONFIG_TRACKER_NODE_ID);
}

