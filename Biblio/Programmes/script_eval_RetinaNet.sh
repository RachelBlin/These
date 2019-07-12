#!/bin/sh
Stokes_weights=`ls /media/rblin/LaCie/weights/Stokes/*.h5`
for weight in $Stokes_weights
alias path_RetinaNet=`cd /home/rblin/Documents/keras-retinanet/`
path_RetinaNet
do
    python keras_retinanet/bin/evaluate.py pascal /home/rblin/Documents/New_polarimetric_DB_3239 test_polar/PARAM_POLAR/Stokes test_polar/LABELS /home/rblin/Documents/weights/Stokes/$Stokes_weights --convert-model
done

