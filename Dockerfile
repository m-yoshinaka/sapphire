FROM python:3.7

WORKDIR /work

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python setup.py develop

CMD [ "/bin/bash" ]
