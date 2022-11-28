import subprocess
import os

logfilename = 'router.log'
print(f"Logging router output to {logfilename}")

with open(logfilename, "w") as f:
    # if not os.path.exists('.crossbar'):
    #     print("Creating router configuration...")
    #     p = subprocess.Popen(['crossbar', 'init'], stdout=f)
    #     p.wait()
    #     print("Done")

    print("Starting service...")
    p = subprocess.Popen(["crossbar", "start", ], stdout=f)
    p.wait()
    print("Shutdown")