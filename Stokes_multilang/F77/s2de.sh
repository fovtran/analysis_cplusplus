#!/bin/bash
#
mkdir temp
cd temp
rm *
~/binc/$ARCH/f77split ../s2de.f
#
for FILE in `ls -1 *.f`;
do
  gfortran -c $FILE
  if [ $? -ne 0 ]; then
    echo "Errors compiling " $FILE
    exit
  fi
done
rm *.f
#
ar qc libs2de.a *.o
rm *.o
#
mv libs2de.a ~/libf77/$ARCH
cd ..
rmdir temp
#
echo "Library installed as ~/libf77/$ARCH/libs2de.a"
