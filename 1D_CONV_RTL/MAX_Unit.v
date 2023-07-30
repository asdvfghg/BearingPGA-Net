module MAX_Unit(numA,numB,Maxout);
    parameter DATA_WIDTH = 16;
    input  [DATA_WIDTH-1:0] numA;
    input  [DATA_WIDTH-1:0] numB;
    output wire  [DATA_WIDTH-1:0] Maxout;
    
//wire [DATA_WIDTH-1:0] addresult;

assign Maxout = (numA[15]^numB[15])?((numA[15]==1&&numB[15]==0)?numB:numA):((numA[15]==1&&numB[15]==1)?(16'b0):((numA[14:0]>numB[14:0])?numA:numB));
//always@(*) begin
//    if(numA[15]==0&&numB[15]==1)
//        Maxout = numA;
//    else if(numA[15]==1&&numB[15]==0)
//        Maxout = numB;
//    else if(numA[15]==0&&numB[15]==0) begin
//            if(numA[14:0]>numB[14:0])
//                 Maxout = numA;
//            else if(numA[14:0]<numB[14:0])
//                 Maxout = numB;
//            else if(numA[14:0]==numB[14:0])
//                 Maxout = numA;
//    end
//    else if(numA[15]==1&&numB[15]==1) begin
//            if(numA[14:0]>numB[14:0])
//                 Maxout = numB;
//            else if(numA[14:0]<numB[14:0])
//                 Maxout = numA;
//            else if(numA[14:0]==numB[14:0])
//                 Maxout = numB;
//    end
//end        
endmodule
