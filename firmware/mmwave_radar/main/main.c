/**
 * @file main.c
 * @brief Plan 2: ESP32 + 60GHz mmWave Radar Fall Detection Gateway
 * 
 * Communicates with mmWave radar via UART, parses posture/fall telemetry,
 * and reports events to Home Assistant / Central Hub via UDP/MQTT.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "lwip/sockets.h"
#include "mdns.h"
#include "mmwave_parser.h"

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

static const char *TAG = "MMWAVE_GATEWAY";

#define UART_NUM UART_NUM_1
#define TXD_PIN 17
#define RXD_PIN 16
#define RX_BUF_SIZE 1024

#define DEST_PORT 5556
#define DEST_IP "255.255.255.255"  // Broadcast fallback; hub binds on 0.0.0.0:5556
#define MDNS_HUB_HOST "falldetect-hub"

static int s_udp_sock = -1;
static struct sockaddr_in s_dest_addr;

static void resolve_hub_mdns(void) {
    esp_err_t err = mdns_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "mDNS init failed (%s), defaulting to broadcast %s", esp_err_to_name(err), DEST_IP);
        return;
    }

    ESP_LOGI(TAG, "Resolving hub IP via mDNS for '%s.local'...", MDNS_HUB_HOST);
    esp_ip4_addr_t addr;
    addr.addr = 0;

    err = mdns_query_a(MDNS_HUB_HOST, 2500, &addr);
    if (err == ESP_OK && addr.addr != 0) {
        s_dest_addr.sin_addr.s_addr = addr.addr;
        ESP_LOGI(TAG, "mDNS successfully resolved %s to " IPSTR " (unicast mode)", MDNS_HUB_HOST, IP2STR(&addr));
    } else {
        ESP_LOGW(TAG, "mDNS query timed out; using broadcast: %s", DEST_IP);
    }
}

static void on_radar_event(const radar_telemetry_t *t) {
    ESP_LOGI(TAG, "[Radar Event] FallState=%d, Posture=%d, Height=%.2fm, Presence=%d, Dwell=%lds, Area=%.2fm2",
             t->fall_state, t->posture, t->target_height_m, t->presence_detected, t->dwell_time_sec, t->cluster_area_m2);

    if (s_udp_sock >= 0) {
        char msg[192];
        int len = snprintf(msg, sizeof(msg),
                           "{\"fall\":%d,\"posture\":%d,\"height\":%.2f,\"dwell\":%ld,\"cluster_area\":%.2f,\"fw_ver\":\"3.1.0\"}\n",
                           t->fall_state, t->posture, t->target_height_m, t->dwell_time_sec, t->cluster_area_m2);
        sendto(s_udp_sock, msg, len, 0, (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
    }
}

static void uart_rx_task(void *arg) {
    uint8_t data[128];
    while (1) {
        int len = uart_read_bytes(UART_NUM, data, sizeof(data), 20 / portTICK_PERIOD_MS);
        if (len > 0) {
            mmwave_parser_feed_buffer(data, (uint16_t)len);
        }
    }
}

static void init_uart(void) {
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM, RX_BUF_SIZE * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_NUM, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_LOGI(TAG, "UART initialized on TX:%d, RX:%d at 115200 baud", TXD_PIN, RXD_PIN);
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

    ESP_LOGI(TAG, "UDP socket ready, streaming radar events to port %d", DEST_PORT);
}

static void init_wifi(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi initialized in STA mode for mmWave Gateway");
}

#if CONFIG_OTA_ENABLED
static void check_and_apply_ota(void) {
    char ota_url[128];
    snprintf(ota_url, sizeof(ota_url),
             "http://" IPSTR ":%d/firmware/radar_v" CONFIG_OTA_VERSION ".bin",
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

    init_wifi();
    init_udp_socket();
#if CONFIG_OTA_ENABLED
    check_and_apply_ota();
#endif
    mmwave_parser_init(on_radar_event);
    init_uart();

    // Create background task to continuously parse UART packets
    xTaskCreate(uart_rx_task, "uart_rx_task", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "mmWave radar gateway running.");
}
