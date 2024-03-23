#!/bin/bash

# Gradescope doesn't allow submitting code from sub-directories.
# This simplifies the process of checking out into a different branch,
# purging everything else, and moving the desired contents up to root.
#
# ex: `bash deploy.sh progset2-submission progset2`

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
	git clean -fd &&
	mv "$2"/* . &&
	rm -r $2
