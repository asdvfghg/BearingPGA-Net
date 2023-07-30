module processingElement16(clk,reset,floatA,floatB,result);

parameter DATA_WIDTH = 16;
parameter RIGHT_SHIFT_BIT = 13;
input clk, reset;
input [DATA_WIDTH-1:0] floatA, floatB;
output reg [DATA_WIDTH-1:0] result;

wire [DATA_WIDTH-1:0] multResult;
wire [DATA_WIDTH-1:0] addResult;

fixedMult16 #(.RIGHT_SHIFT_BIT(RIGHT_SHIFT_BIT))FM (floatA,floatB,multResult);
fixedAdd16 FADD(multResult,result,addResult);


always @ (posedge clk) begin
	if (reset == 1'b0) begin
		result <= 0;
	end else begin
		result <= addResult;
	end 
end

endmodule
