module shiftFunction(clk,reset,input_relu,output_fc, done_flag);

parameter DATA_WIDTH = 16;
parameter OUTPUT_NODES = 256;
parameter SHIFT_BIT = 5;
input clk, reset;
input [DATA_WIDTH*OUTPUT_NODES-1:0] input_relu;
output reg [DATA_WIDTH*OUTPUT_NODES-1:0] output_fc;
output reg done_flag;

reg [9:0] i;

always @ (posedge clk) begin
	if (reset == 1'b0) begin
		output_fc <= 0;
	    done_flag <= 1'b0;
	end else begin
			for (i = 0; i < OUTPUT_NODES; i = i + 1) begin
					output_fc[DATA_WIDTH*i+:DATA_WIDTH] <= {input_relu[DATA_WIDTH*i-1+DATA_WIDTH],{SHIFT_BIT{1'b0}},input_relu[DATA_WIDTH*i+SHIFT_BIT +:11]};
			end
	     if(i == OUTPUT_NODES) begin
	           done_flag <= 1'b1;
	     end
	end
end

endmodule
