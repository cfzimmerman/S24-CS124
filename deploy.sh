#!/bin/bash

git checkout $1 
  && cd $2 && cd .. # verify that arg 2 is a valid directory
  && rm -rf (ls | grep -v $2) 
  && cd $2 
  && mv . ..
  && cd ..
  && rmdir $2

