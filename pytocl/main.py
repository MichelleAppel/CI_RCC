"""Application entry point."""
import argparse
import logging
import multiprocessing
from pytocl.protocol import Client


def main(driver, NO_FRAMES, PORT, que, print_car_values=False):
    """Main entry point of application."""
    parser = argparse.ArgumentParser(
        description='Client for TORCS racing car simulation with SCRC network'
                    ' server.'
    )
    parser.add_argument(
        '--hostname',
        help='Racing server host name.',
        default='localhost'
    )
    parser.add_argument(
        '-p',
        '--port',
        help='Port to connect, 3001 - 3010 for clients 1 - 10.',
        type=int,
        default=PORT
    )
    parser.add_argument('-v', help='Debug log level.', action='store_true')
    args = parser.parse_args()

    # switch log level:
    if args.v:
        level = logging.DEBUG
    else:
        level = logging.INFO
    del args.v
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)7s %(name)s %(message)s"
    )

    # start client loop:
    client = Client(driver=driver, **args.__dict__)
    fitness, offroad_count, turn_around_count, negative_speed_count = client.run(NO_FRAMES)
    que.put(fitness)

if __name__ == '__main__':
    from pytocl.driver import Driver

    main(Driver())
