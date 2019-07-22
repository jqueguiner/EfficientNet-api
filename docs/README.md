# Docker for API

You can build and run the docker using the following process:

Cloning
```console
git clone REPO 
```

Building Docker
```console
cd  && docker build -t  -f Dockerfile .
```

Running Docker
```console
echo "http://$(curl ifconfig.io):5000" && docker run -p 5000:5000 -d 
```

Calling the API
```console
CALL
```