import subprocess
import os

logfilename = 'router.log'
print(f"Logging router output to {logfilename}")

def init_config(logger):
    if not os.path.isdir('./.crossbar'):
        print("Initializing router configuration...")
        p = subprocess.Popen(["crossbar", "init", ], stdout=logger)
        p.wait()
        import json
        print("Changing configuration...")
        with open('./.crossbar/config.json', 'r') as f:
            cfg = json.load(f)
        cfg['workers'][0]['realms'][0]['name'] = 'adviser'
        with open('./.crossbar/config.json', 'w') as f:
            json.dump(cfg, f)
        print("Done")


with open(logfilename, "w") as f:
    # if not os.path.exists('.crossbar'):
    #     print("Creating router configuration...")
    #     p = subprocess.Popen(['crossbar', 'init'], stdout=f)
    #     p.wait()
    #     print("Done")

    init_config(f)
    print("Starting service...")
    p = subprocess.Popen(["crossbar", "start", ], stdout=f)
    p.wait()
    print("Shutdown")