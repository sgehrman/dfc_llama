#!/bin/bash

flutter pub upgrade  

cd ./example
flutter pub upgrade  
cd $OLDPWD

echo '## all done'
