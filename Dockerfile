FROM python:3.7

RUN echo "[FreeTDS]\n\
Description = FreeTDS unixODBC Driver\n\
Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\n\
Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini

RUN export PYMSSQL_BUILD_WITH_BUNDLED_FREETDS=1
RUN apt-get update && apt-get install -y unixodbc unixodbc-dev freetds-dev freetds-bin tdsodbc

USER root
WORKDIR /app
add . /app
RUN pip install -r /app/requirements.txt
EXPOSE 8080:8080
RUN pwd
RUN chmod +x start.sh
CMD ["/bin/bash", "start.sh"]
