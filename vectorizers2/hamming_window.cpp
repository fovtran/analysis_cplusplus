for(int i = 0; i < SEGMENTATION_LENGTH;i++){
    timeDomain[i] = (float) (( 0.53836 - ( 0.46164 * Math.cos( TWOPI * (double)i  / (double)( SEGMENTATION_LENGTH - 1 ) ) ) ) * frameBuffer[i]);
}