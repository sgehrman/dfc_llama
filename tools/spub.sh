#!/bin/bash

flutter pub upgrade --major-versions --tighten

cd ./example
flutter pub upgrade --major-versions --tighten
cd $OLDPWD

echo '## all done'
