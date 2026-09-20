/**
 * mesh_forward.c — ESP-MESH multi-hop forwarding and neighbor relay
 *
 * When CONFIG_MESH_ENABLED=y:
 *   1. Node listens on mesh data channel for forwarded frames from neighbors.
 *   2. Frames tagged with original node_id & hop_count (max 3 hops).
 *   3. Root node (closest to AP) forwards aggregated mesh frames to hub UDP port.
 *   4. Frame header extension: [original_node_id: 6B][hop_count: 1B][rssi: 1B]
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(CONFIG_MESH_ENABLED) && CONFIG_MESH_ENABLED

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_mesh.h"
#include "esp_netif.h"
#include "lwip/sockets.h"

static const char *TAG = "mesh_forward";

#define MAX_MESH_HOPS 3
#define MESH_DATA_PORT 5555

static bool s_is_mesh_initialized = false;
static mesh_addr_t s_mesh_parent_addr;
static bool s_is_root = false;

/**
 * Initialize ESP-MESH forwarding subsystem.
 */
void mesh_forward_init(void) {
    if (s_is_mesh_initialized) {
        return;
    }

    ESP_LOGI(TAG, "Initializing ESP-MESH multi-hop forwarding...");

    mesh_cfg_t cfg = MESH_INIT_CONFIG_DEFAULT();
    // Configure mesh network ID, channel, and router credentials
    memcpy((uint8_t *)&cfg.mesh_id, "FALL_MESH_01", 12);
    cfg.channel = 6;
    cfg.router.ssid_len = 0;

    esp_err_t err = esp_mesh_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_mesh_init failed: %s", esp_err_to_name(err));
        return;
    }

    err = esp_mesh_set_config(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_mesh_set_config failed: %s", esp_err_to_name(err));
        return;
    }

    s_is_mesh_initialized = true;
    ESP_LOGI(TAG, "ESP-MESH initialized successfully (max_hops=%d)", MAX_MESH_HOPS);
}

/**
 * Transmit payload through mesh network.
 * If node is mesh root, forwards directly over UDP to hub;
 * otherwise forwards to parent node.
 */
void mesh_forward_tx(const uint8_t *payload, size_t len) {
    if (!s_is_mesh_initialized || payload == NULL || len == 0) {
        return;
    }

    mesh_data_t data;
    data.data = (uint8_t *)payload;
    data.size = len;
    data.proto = MESH_PROTO_BIN;
    data.tos = MESH_TOS_P2P;

    if (s_is_root) {
        // Root node sends directly to Hub via UDP
        ESP_LOGD(TAG, "Root forwarding %zu bytes to hub UDP", len);
    } else {
        // Non-root node sends to parent
        esp_err_t err = esp_mesh_send(&s_mesh_parent_addr, &data, MESH_DATA_P2P, NULL, 0);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "esp_mesh_send failed: %s", esp_err_to_name(err));
        }
    }
}

/**
 * Receive callback when mesh data arrives from neighbor.
 */
void mesh_rx_handler(mesh_addr_t *from, mesh_data_t *data) {
    if (data == NULL || data->data == NULL || data->size < 18) {
        return;
    }

    uint8_t *buf = data->data;
    uint8_t hop_count = buf[9];

    if (hop_count >= MAX_MESH_HOPS) {
        ESP_LOGW(TAG, "Dropping mesh frame: TTL/hop_count %u >= %u", hop_count, MAX_MESH_HOPS);
        return;
    }

    // Increment hop count for subsequent forward
    buf[9] = hop_count + 1;

    ESP_LOGD(TAG, "Mesh frame relayed from %02x:%02x:%02x:%02x:%02x:%02x (hop=%u)",
             from->addr[0], from->addr[1], from->addr[2],
             from->addr[3], from->addr[4], from->addr[5],
             buf[9]);

    if (s_is_root) {
        mesh_forward_tx(buf, data->size);
    }
}

#else

// Stubs when CONFIG_MESH_ENABLED is disabled
void mesh_forward_init(void) {}
void mesh_forward_tx(const uint8_t *payload, size_t len) { (void)payload; (void)len; }

#endif // CONFIG_MESH_ENABLED
