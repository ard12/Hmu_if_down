/**
 * @file mmwave_parser.h
 * @brief Binary frame parser for 60 GHz mmWave Fall Detection Radar (MR60FDA1/HLK)
 */

#ifndef MMWAVE_PARSER_H
#define MMWAVE_PARSER_H

#include <stdint.h>
#include <stdbool.h>

#define MMWAVE_FRAME_HEADER1 0x53
#define MMWAVE_FRAME_HEADER2 0x59

typedef enum {
    RADAR_FALL_NONE = 0,
    RADAR_FALL_SUSPECTED = 1,
    RADAR_FALL_CONFIRMED = 2,
    RADAR_FALL_DWELL_FLOOR = 3,
} radar_fall_state_t;

typedef enum {
    RADAR_POSTURE_UNKNOWN = 0,
    RADAR_POSTURE_STANDING = 1,
    RADAR_POSTURE_SITTING = 2,
    RADAR_POSTURE_LYING = 3,
} radar_posture_t;

typedef struct {
    radar_fall_state_t fall_state;
    radar_posture_t posture;
    float target_height_m;
    float target_distance_m;
    bool presence_detected;
    uint32_t dwell_time_sec;
    float cluster_area_m2;
    char firmware_version[16];
} radar_telemetry_t;

typedef void (*radar_event_callback_t)(const radar_telemetry_t *telemetry);

void mmwave_parser_init(radar_event_callback_t callback);
void mmwave_parser_feed_byte(uint8_t byte);
void mmwave_parser_feed_buffer(const uint8_t *buf, uint16_t len);

#endif // MMWAVE_PARSER_H
