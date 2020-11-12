import argparse


p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument("--param1", help="First prameter")
p.add_argument("--param2", help="Second parameter")
p.add_argument(
    "--combined",
    help="A param that will come from two dataset attributes")
p.add_argument("--run", "-r", help="run")


args, extra = p.parse_known_args()

print("Run:{}, P1:{}, P2:{}, C:{}, extras:{}".format(
    args.run, args.param1, args.param2, args.combined, extra))
