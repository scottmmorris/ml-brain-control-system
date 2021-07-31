from eeg_subscribe import Cortex

class Subcribe():
	def __init__(self):
		self.c = Cortex(user, debug_mode=True)
		self.c.do_prepare_steps()

	def sub(self, streams):
		self.c.sub_request(streams)

user = {
	"license" : "license",
	"client_id" : "client_id",
	"client_secret" : "client_secret",
}


s = Subcribe()

streams = ['eeg']

s.sub(streams)

# -----------------------------------------------------------

# -----------------------------------------------------------
# 
# SETTING
# 	- replace your license, client_id, client_secret to user dic
# 	- specify infor for record and export
# 	- connect your headset with dongle or bluetooth, you should saw headset on EmotivApp
#
# 
# RESULT
# 	- subcribed data type should print out to console log
# 	- "cols" contain order and column name of output data
# 
# 
#	{"id":6,"jsonrpc":"2.0","result":{"failure":[],"success":[{"cols":["COUNTER","INTERPOLATED","T7","T8","RAW_CQ","MARKER_HARDWARE","MARKERS"],"sid":"0fd1c571-f0ec-4aa0-bb71-b4fa2b9c7504","streamName":"eeg"}]}}
# 	{"eeg":[4,0,4222.476,4202.952,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6683}
# 	{"eeg":[5,0,4220.571,4204.857,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6761}
# 	{"eeg":[6,0,4219.143,4207.238,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6839}
# 	{"eeg":[7,0,4218.667,4198.667,0.0,0,[]],"sid":"866e47d8-d7e6-4cfa-87b7-c4f956d6c429","time":1590984953.6917}

# user = {
# 	"license" : "7238fd0a-bbf2-4578-b1d2-cd674d5104d3",
# 	"client_id" : "ehbAwt6qQhv10IuN8kkOWzzRHXLKqbAS4EoGKLHj",
# 	"client_secret" : "UplxIH9ltrYKYYbkYmvXDuLWcUTGMZDbDzCxbVh6sLiWptCXI9NSy03257xSBv7JlH531Ws3TTHxmMUyevxjbItJcIvrg6IHfRnG6dyyDjJk6U2bKLc9gnjDFDlUFCFq",
# }
# -----------------------------------------------------------