/**
 * @file main.c
 * @brief Node 0: Wi-Fi CSI Active Transmitter (AP / ESP-NOW Beacon Injector)
 * 
 * Periodically broadcasts 16-byte ESP-NOW ping frames at 100 Hz (every 10ms)
 * to provide a deterministic, high-rate RF carrier for Tracker Nodes 1, 2, and 3.
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "nvs_flash.h"

static const char *TAG = "CSI_TRANSMITTER";

#define WIFI_CHANNEL 6
#define INJECTION_PERIOD_US 10000 // 10 ms = 100 Hz

// Broadcast MAC address
static const uint8_t s_broadcast_mac[ESP_NOW_ETH_ALEN] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// 16-byte packet payload with magic header and sequence counter
typedef struct __attribute__((packed)) {
    uint8_t magic[4];       // "FALL"
    uint32_t seq_num;
    uint32_t timestamp_ms;
    uint32_t reserved;
} csi_ping_packet_t;

static csi_ping_packet_t s_packet;
static esp_timer_handle_t s_periodic_timer;

static void periodic_timer_callback(void *arg) {
    s_packet.seq_num++;
    s_packet.timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000);

    esp_err_t err = esp_now_send(s_broadcast_mac, (uint8_t *)&s_packet, sizeof(s_packet));
    if (err != ESP_OK) {
        ESP_LOGD(TAG, "ESP-NOW send failed: %d", err);
    }
}

static void init_wifi(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Set dedicated Wi-Fi channel
    ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE));
    ESP_LOGI(TAG, "Wi-Fi initialized on Channel %d", WIFI_CHANNEL);
}

static void init_esp_now(void) {
    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer_info = {0};
    memcpy(peer_info.peer_addr, s_broadcast_mac, ESP_NOW_ETH_ALEN);
    peer_info.channel = WIFI_CHANNEL;
    peer_info.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer_info));

    ESP_LOGI(TAG, "ESP-NOW initialized with broadcast peer");
}

void app_main(void) {
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    init_wifi();
    init_esp_now();

    // Prepare packet template
    memcpy(s_packet.magic, "FALL", 4);
    s_packet.seq_num = 0;

    // Start 100 Hz high-resolution timer
    const esp_timer_create_args_t timer_args = {
        .callback = &periodic_timer_callback,
        .name = "csi_ping_timer"
    };
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &s_periodic_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(s_periodic_timer, INJECTION_PERIOD_US));

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Transmitter MAC Address: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    ESP_LOGI(TAG, "Started 100 Hz active injection timer. Node 0 AP is running.");
}
