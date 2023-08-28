from middlewordextractor import MiddleWordExtractor
import pandas as pd

data = pd.read_json("data.json")

mwext = MiddleWordExtractor(data)

print(mwext.forward("I want simulation services at AR, VR, through metaverse."))