module FcBiasAdd(fc_in,biases,cnn_out);
//reset,
parameter DATA_WIDTH = 16;
parameter OUTPUT_NODES = 10;


input [DATA_WIDTH*OUTPUT_NODES-1:0] fc_in;//160
input [DATA_WIDTH*OUTPUT_NODES-1:0] biases;//10*16=160
output [DATA_WIDTH*OUTPUT_NODES-1:0] cnn_out;//10*16=160

reg [DATA_WIDTH-1:0] selectedInput;

genvar i;
generate
	for (i = 0; i < OUTPUT_NODES; i = i + 1) begin//执行10次
        fixedAdd16 FADD
        (
            .a(fc_in[DATA_WIDTH*i+:DATA_WIDTH]),
            .b(biases[DATA_WIDTH*i+:DATA_WIDTH]),
            .result(cnn_out[DATA_WIDTH*i+:DATA_WIDTH])
         );
	end
endgenerate

//integer j;
//always @ (posedge clk or negedge reset) begin//
//	if (reset == 1'b0) begin
//		selectedInput = 0;
//		j = OUTPUT_NODES - 1;//511
//	end else if (j < 0) begin
//		selectedInput = 0;
//	end else begin
//		selectedInput = fc_in[DATA_WIDTH*j+:DATA_WIDTH];//（高16位）
//		j = j - 1;
//	end
//end

endmodule
