import argparse
import json


def main():
    parser=argparse.ArgumentParser(description="Build the offline Bitcoin sentiment case study")
    parser.add_argument("command",choices=["fetch","curate","model","score","audit","audit-model"])
    args=parser.parse_args()
    if args.command in ["fetch","curate"]:
        from .data import download_sources,curate
        if args.command=="fetch":
            download_sources()
        result=curate()
    elif args.command=="model":
        from .sentiment import download_model
        result=download_model()
    else:
        from .pipeline import build,audit
        result=build() if args.command=="score" else audit(args.command=="audit-model")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
