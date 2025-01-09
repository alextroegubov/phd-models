from argparse import ArgumentParser


import analytical
import simulation



def get_arg_parser():
    pass
    parser = ArgumentParser(
        prog="",
        description=""
    )

    # real-time data
    parser.add_argument('--rt_n', type=int, help='Number of real-time data flows')
    parser.add_argument('--rt_lambda', nargs='+', help="Real-time data flows arrival rates")
    parser.add_argument('--rt_mu', type=float, nargs='+', help="Real-time data flows service rate")
    parser.add_argument('--rt_b', type=float, nargs='+', help="Real-time data flows resource size")

    # elastic data
    parser.add_argument('--data_b_min', type=int, help="Minimum resource")
    parser.add_argument('--data_b_max', type=int, help="Maximum resource")
    parser.add_argument('--data_lambda', type=float, help="Data flow arrival rate")
    parser.add_argument('--data_mu', type=float, help="Data flow service rate")

    # general
    parser.add_argument('--capacity', type=int, help="Beam capacity")
    
    # analytical
    parser.add_argument()
