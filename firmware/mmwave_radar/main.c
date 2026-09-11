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
#include "mmwave_parser.h"

static const char *TAG = "MMWAVE_GATEWAY";

#define UART_NUM UART_NUM_1
#define TXD_PIN 17
#define RXD_PIN 16
#define RX_BUF_SIZE 1024

#define DEST_PORT 5556
#define DEST_IP "192.168.4.2"

static int s_udp_sock = -1;
static struct sockaddr_in s_dest_addr;

static void on_radar_event(const radar_telemetry_t *t) {
    ESP_LOGI(TAG, "[Radar Event] FallState=%d, Posture=%d, Height=%.2fm, Presence=%d, Dwell=%lds",
             t->fall_state, t->posture, t->target_height_m, t->presence_detected, t->dwell_time_sec);

    if (s_udp_sock >= 0) {
        char msg[128];
        int len = snprintf(msg, sizeof(msg),
                           "{\"fall\":%d,\"posture\":%d,\"height\":%.2f,\"dwell\":%ld}\n",
                           t->fall_state, t->posture, t->target_height_m, t->dwell_time_sec);
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

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    mmwave_parser_init(on_radar_event);
    init_uart();

    // Create background task to continuously parse UART packets
    xTaskCreate(uart_rx_task, "uart_rx_task", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "mmWave radar gateway running.");
}
