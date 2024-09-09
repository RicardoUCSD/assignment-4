__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth,  int imageChannels){
    
    //@@ Insert code to implement matrix multiplication here
    int maskRadius = maskWidth/2; //this is integer division, so the result is 2
 
    int i = get_global_id(0); //height
    int j = get_global_id(1); //width 
        
    for (int k=0;k<imageChannels;k++){ //Itherate through multiplications
        float accum=0;
        //Loop over y pixels
        for (int y =-maskRadius;y<=maskRadius;y++){      //->y
            for (int x =-maskRadius;x<=maskRadius;x++){  //->x
                int xOffset = j+x;
                int yOffset = i+y;
                //take care of image bounds/ edges, pixels value=0
                if ( ((xOffset>=0) && (xOffset<width)) && ((yOffset>=0) && (yOffset<height)) ){ 
                    float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                    float maskValue = maskData[(y + maskRadius) * maskWidth + x + maskRadius];
                    accum += imagePixel * maskValue;
                }
            }
        }
        accum = (accum<0)?0:(accum>1)?1:accum; //clamping, pixels are in the range of 0 to 1
        outputData[(i * width + j) * imageChannels + k] = accum;
    }
}