`timescale 1 ns / 10 ps

module maxPoolSingle(mPoolIn,mPoolOut);
  
parameter DATA_WIDTH = 16;
parameter InputW = 256;
parameter Depth = 1;

input [0:InputW*Depth*DATA_WIDTH-1] mPoolIn;
output [0:(InputW/2)*Depth*DATA_WIDTH-1] mPoolOut;

genvar i,j;

generate 
    for (i=0; i<(InputW); i=i+2) begin
    MAX_Unit
    #(
     .DATA_WIDTH(DATA_WIDTH)
     )
     MU
     (
     .numA(mPoolIn[i*DATA_WIDTH+:DATA_WIDTH]),
     .numB(mPoolIn[(i+1)*DATA_WIDTH+:DATA_WIDTH]),
     .Maxout(mPoolOut[(i/2)*DATA_WIDTH+:DATA_WIDTH])
     );
  end
endgenerate

endmodule
      