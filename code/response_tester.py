import requests

# URL of the JSON file
url = 'http://192.168.0.102:8000/'

while True:
	# Send an HTTP GET request to the URL
	try:
		response = requests.get(url)
		# Check if the request was successful
		if response.status_code == 200:
			# Parse the JSON content
			data = response.json()
			print(data)
		else:
			print(f'Failed to retrieve data: {response.status_code}')
		
	except Exception as e: print(e)
