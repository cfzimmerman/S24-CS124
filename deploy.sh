#!/bin/bash

if [ "$1" = "main" ]; then
	echo "Cannot deploy from main"
	exit 1
fi

git checkout $1 &&
	git fetch origin &&
	cd $2 && # verify that arg 2 is a valid directory
	cd .. &&
	git reset --hard origin/main &&
	rm -rf $(ls | grep -v $2) &&
	cd $2 &&
	mv . .. &&
	cd .. &&
	rmdir $2
