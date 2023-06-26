# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 23:12:56 2022

@author: MIR TAQI MIR
"""
import json
import requests
resp= requests.get("https://stackoverflow.com/")
resp_dict = json.loads(resp.text)
print(resp_dict)


servicePlanId = ""
apiToken = ""
sinchNumber = ""
toNumber = ""
url = "https://us.sms.api.sinch.com/xms/v1/" + servicePlanId + "/batches"

payload = {
  "from": sinchNumber,
  "to": [
    toNumber
  ],
  "body": "Hello how are you"
}

headers = {
  "Content-Type": "application/json",
  "Authorization": "Bearer " + apiToken
}

response = requests.post(url, json=payload, headers=headers)

data = response.json()
print(data)