# TODO
# Receive mission phase from ground control
# Match mission phase to entry point
# Phase 1: Take off, orient to target, and fly to target N & E
# Phase 2: Ensure we're within target radius, initiate land
# Phase 3: Ensure megadrone coordinates are stable for more than 5 seconds and is in a disarmed state, then initiate takeoff
# Phase 4: Fly to megadrone N & E
# Phase 5: Loop until we see the landing apriltag ID beneath the DEXI, once we see the landing apriltag ID for more than 5 seconds, allow phase 6
# Phase 6: Switch to apriltag landing mode and orient to apriltag, then land


def main():
    print('Hi from worker_mission_control.')


if __name__ == '__main__':
    main()
