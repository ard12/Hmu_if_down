/**
 * @file mmwave_parser.c
 * @brief Implementation of 60 GHz mmWave Radar frame state machine
 */

#include "mmwave_parser.h"
#include <string.h>

typedef enum {
    STATE_WAIT_HEADER1,
    STATE_WAIT_HEADER2,
    STATE_WAIT_CTRL,
    STATE_WAIT_CMD,
    STATE_WAIT_LEN_H,
    STATE_WAIT_LEN_L,
    STATE_WAIT_DATA,
    STATE_WAIT_CHECKSUM,
} parser_state_t;

#define MAX_PAYLOAD_LEN 64

static parser_state_t s_state = STATE_WAIT_HEADER1;
static uint8_t s_ctrl = 0;
static uint8_t s_cmd = 0;
static uint16_t s_data_len = 0;
static uint16_t s_data_index = 0;
static uint8_t s_payload[MAX_PAYLOAD_LEN];
static uint8_t s_running_checksum = 0;

static radar_event_callback_t s_event_cb = NULL;
static radar_telemetry_t s_current_telemetry;

void mmwave_parser_init(radar_event_callback_t callback) {
    s_event_cb = callback;
    s_state = STATE_WAIT_HEADER1;
    memset(&s_current_telemetry, 0, sizeof(s_current_telemetry));
}

static void process_frame(void) {
    bool updated = false;

    // Control Word 0x02: Fall Detection and State
    if (s_ctrl == 0x02) {
        if (s_cmd == 0x01 && s_data_len >= 1) { // Fall status
            s_current_telemetry.fall_state = (radar_fall_state_t)s_payload[0];
            updated = true;
        } else if (s_cmd == 0x02 && s_data_len >= 2) { // Fall dwell duration
            s_current_telemetry.dwell_time_sec = (s_payload[0] << 8) | s_payload[1];
            updated = true;
        }
    }
    // Control Word 0x03: Posture and 3D Altitude
    else if (s_ctrl == 0x03) {
        if (s_cmd == 0x01 && s_data_len >= 1) { // Posture status
            s_current_telemetry.posture = (radar_posture_t)s_payload[0];
            updated = true;
        } else if (s_cmd == 0x02 && s_data_len >= 2) { // Target Centroid Height (cm)
            uint16_t height_cm = (s_payload[0] << 8) | s_payload[1];
            s_current_telemetry.target_height_m = (float)height_cm / 100.0f;
            updated = true;
        }
    }
    // Control Word 0x04: Presence detection
    else if (s_ctrl == 0x04) {
        if (s_cmd == 0x01 && s_data_len >= 1) {
            s_current_telemetry.presence_detected = (s_payload[0] != 0);
            updated = true;
        }
    }

    if (updated && s_event_cb) {
        s_event_cb(&s_current_telemetry);
    }
}

void mmwave_parser_feed_byte(uint8_t byte) {
    switch (s_state) {
        case STATE_WAIT_HEADER1:
            if (byte == MMWAVE_FRAME_HEADER1) {
                s_running_checksum = byte;
                s_state = STATE_WAIT_HEADER2;
            }
            break;

        case STATE_WAIT_HEADER2:
            if (byte == MMWAVE_FRAME_HEADER2) {
                s_running_checksum += byte;
                s_state = STATE_WAIT_CTRL;
            } else {
                s_state = STATE_WAIT_HEADER1;
            }
            break;

        case STATE_WAIT_CTRL:
            s_ctrl = byte;
            s_running_checksum += byte;
            s_state = STATE_WAIT_CMD;
            break;

        case STATE_WAIT_CMD:
            s_cmd = byte;
            s_running_checksum += byte;
            s_state = STATE_WAIT_LEN_H;
            break;

        case STATE_WAIT_LEN_H:
            s_data_len = ((uint16_t)byte) << 8;
            s_running_checksum += byte;
            s_state = STATE_WAIT_LEN_L;
            break;

        case STATE_WAIT_LEN_L:
            s_data_len |= byte;
            s_running_checksum += byte;
            s_data_index = 0;
            if (s_data_len > MAX_PAYLOAD_LEN) {
                s_state = STATE_WAIT_HEADER1; // Buffer overflow safeguard
            } else if (s_data_len == 0) {
                s_state = STATE_WAIT_CHECKSUM;
            } else {
                s_state = STATE_WAIT_DATA;
            }
            break;

        case STATE_WAIT_DATA:
            s_payload[s_data_index++] = byte;
            s_running_checksum += byte;
            if (s_data_index >= s_data_len) {
                s_state = STATE_WAIT_CHECKSUM;
            }
            break;

        case STATE_WAIT_CHECKSUM:
            if (byte == s_running_checksum) {
                process_frame();
            }
            s_state = STATE_WAIT_HEADER1;
            break;
    }
}

void mmwave_parser_feed_buffer(const uint8_t *buf, uint16_t len) {
    for (uint16_t i = 0; i < len; i++) {
        mmwave_parser_feed_byte(buf[i]);
    }
}
