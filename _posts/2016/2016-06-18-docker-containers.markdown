---
layout: post
title: "Docker Containers"
date: "2016-06-18 17:53"
---

Docker is an open source platform which can be used to package, distribute and run your applications. Docker provides an easy and efficient way to encapsulate applications (e.g. a Java web application) and any required infrastructure to run that application (e.g. Red hat Linux OS, Apache web server, Tomcat application server, mySQL database etc.) as a single “Docker image” which can then be shared through a central, shared “Docker registry“. The image can then be used to launch a “Docker container” which makes the contained application available from the host where the Docker container is running.

Docker provides some convenient tools to build Docker images in a simple and efficient way. A Docker container on the other hand is a kind of light weight virtual machine with considerably smaller memory and disk space footprint than a full blown virtual machine.

![Docker Stack]({{site.url}}images/2016/06/docker.png)
