/**
 * @file main.c
 * @brief Tracker Nodes (Nodes 1, 2, 3): Wi-Fi CSI Receiver and Streamer
 * 
 * Configures CSI collection callback on ESP32/ESP32-C6, extracts raw
 * subcarrier I/Q information, packages with Node ID, and streams via UDP
 * to the central Fall Detection processing hub.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

static const char *TAG = "CSI_TRACKER";

// Configure Node ID: 1, 2, or 3 (change per flashed board)
#ifndef CONFIG_TRACKER_NODE_ID
#define CONFIG_TRACKER_NODE_ID 1
#endif

#define WIFI_CHANNEL 6
#define DEST_PORT 5555
#define DEST_IP "192.168.4.2" // Central Hub IP or broadcast "255.255.255.255"

// Header for UDP CSI Stream Packet
typedef struct __attribute__((packed)) {
    uint8_t magic[4];          // "CSIF"
    uint8_t node_id;           // 1, 2, or 3
    int8_t rssi;               // Packet RSSI in dBm
    uint16_t subcarrier_count; // Number of subcarriers
    uint32_t timestamp_ms;
    uint32_t seq_num;
} csi_udp_header_t;

static int s_udp_sock = -1;
static struct sockaddr_in s_dest_addr;
static uint32_t s_packet_counter = 0;

static void wifi_csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len == 0 || s_udp_sock < 0) {
        return;
    }

    uint16_t subcarrier_count = info->len / 2;
    if (subcarrier_count > 128) {
        subcarrier_count = 128; // Cap for compact UDP frame
    }

    // Prepare packet buffer: Header + I/Q bytes
    uint8_t buffer[sizeof(csi_udp_header_t) + 256];
    csi_udp_header_t *hdr = (csi_udp_header_t *)buffer;

    memcpy(hdr->magic, "CSIF", 4);
    hdr->node_id = (uint8_t)CONFIG_TRACKER_NODE_ID;
    hdr->rssi = info->rx_ctrl.rssi;
    hdr->subcarrier_count = subcarrier_count;
    hdr->timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);
    hdr->seq_num = s_packet_counter++;

    // Copy raw I/Q subcarrier bytes
    memcpy(buffer + sizeof(csi_udp_header_t), info->buf, subcarrier_count * 2);

    int total_len = sizeof(csi_udp_header_t) + (subcarrier_count * 2);
    sendto(s_udp_sock, buffer, total_len, 0, (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
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

    ESP_LOGI(TAG, "UDP socket ready, streaming to %s:%d", DEST_IP, DEST_PORT);
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

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    init_udp_socket();
    init_wifi_csi();

    ESP_LOGI(TAG, "Tracker Node %d is actively running and streaming CSI.", CONFIG_TRACKER_NODE_ID);
}
