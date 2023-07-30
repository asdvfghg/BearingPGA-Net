module FC_weights(clk,address,weights);

parameter DATA_WIDTH = 16;
parameter FC_IN = 256;
parameter FC_OUT = 10;

input clk;
input [8:0] address;
output wire [DATA_WIDTH*FC_OUT-1:0] weights;

fc_weights fcW
( .clka(clk),
  .addra(address),
  .douta(weights));

endmodule

