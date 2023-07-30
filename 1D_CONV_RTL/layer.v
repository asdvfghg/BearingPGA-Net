module layer(clk,reset,input_fc,output_fc,done_flag);//,weights

parameter DATA_WIDTH = 16;
parameter INPUT_NODES = 256;
parameter OUTPUT_NODES = 10;
parameter RIGHT_SHIFT_BIT = 8;
input clk, reset;
input [DATA_WIDTH*INPUT_NODES-1:0] input_fc;//256*16
//input [DATA_WIDTH*OUTPUT_NODES-1:0] weights;//10*16
output  [DATA_WIDTH*OUTPUT_NODES-1:0] output_fc;//10*16
output reg done_flag;

reg [8:0] address;
wire [DATA_WIDTH*OUTPUT_NODES-1:0] weights;//10*16
reg FCstart;

reg [DATA_WIDTH-1:0] selectedInput;
wire [DATA_WIDTH*OUTPUT_NODES-1:0] output_fc_reg;
wire [DATA_WIDTH*OUTPUT_NODES-1:0] FC_B = 160'b0000010000001011100000011101010110000001101101101000010010101000000000011100100100000000111011111000001001001101100000001111111010000100000001110000100010100000;
//1000000000001110_0000000000111100_0000000001000011_1000000000001110_1000000000001001_1000000001000000_1000000001001110_0000000001000110_1000000001011011_0000000001001110;
//1000000100101100_1000001000011101_1000000110001011_1000000110010111_1000000010101011_1000001000101001_1000000111011000_1000000101110101_1000000101101000_1000000011011000;

genvar i;
generate
	for (i = 0; i < OUTPUT_NODES; i = i + 1) begin//执行10次
		processingElement16#(
		.RIGHT_SHIFT_BIT(RIGHT_SHIFT_BIT)
		) PE 
		(
			.clk(clk),
			.reset(FCstart),
			.floatA(selectedInput),
			.floatB(weights[DATA_WIDTH*i+:DATA_WIDTH]),
			.result(output_fc_reg[DATA_WIDTH*i+:DATA_WIDTH])
		);
	end
endgenerate

genvar k;
generate
	for (k = 0; k < OUTPUT_NODES; k = k + 1) begin//执行10次
        fixedAdd16 FADD
        (
            .a(output_fc_reg[DATA_WIDTH*k+:DATA_WIDTH]),
            .b(FC_B[DATA_WIDTH*k+:DATA_WIDTH]),
            .result(output_fc[DATA_WIDTH*k+:DATA_WIDTH])
         );
	end
endgenerate

FC_weights  FCM(
    .clk(clk),
    .address(address),
    .weights(weights));


always @ (posedge clk) begin
	if (reset == 1'b0) begin
		address <= 0;
		done_flag <= 0;
	end
    else begin
	       if (address != 9'd258) 
                address <= address + 1;
           else begin
                address <= 9'd0;
                done_flag <= 1'b1;
           end
        end
end

always@(posedge clk) begin
    if(reset == 0)
        FCstart <= 1'b0;
    else
        FCstart <= 1'b1;
end

reg [8:0] j;
always @ (posedge clk) begin
	if (FCstart == 1'b0) begin
		selectedInput <= 0;
		j <= INPUT_NODES;//255
	end else if (j == 0) begin//j < 0
		selectedInput <= 0;
	end else begin
		selectedInput <= input_fc[DATA_WIDTH*(j-1)+:DATA_WIDTH];//高16位
		j <= j - 1;
	end
end

endmodule
