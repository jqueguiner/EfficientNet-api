# Docker for API

You can build and run the docker using the following process:

Cloning
```console
git clone https://github.com/jqueguiner/EfficientNet-api EfficientNet-api
```

Building Docker
```console
cd EfficientNet-api && docker build -t EfficientNet-api -f Dockerfile .
```

Running Docker
```console
echo "http://$(curl ifconfig.io):5000" && docker run -p 5000:5000 -d EfficientNet-api
```

Calling the API
```console
curl -X POST "http://MY_SUPER_API_IP:5000/detect" -H "accept: application/json" -H "Content-Type: application/json" -d '{"url":"https://i.ibb.co/YBkwZks/input.jpg", "top_k": 5}'
```
