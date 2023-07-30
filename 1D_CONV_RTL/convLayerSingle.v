`timescale 100 ns / 10 ps

module convLayerSingle(clk,reset,signal,filter,bias,outputConv);

parameter DATA_WIDTH = 16;
parameter D = 1; //Depth of the filter
parameter H = 1; //Height of the image
parameter W = 1024; //Width of the image
parameter F = 64; //Size of the filter
parameter P = 28;//padding of image
parameter S = 8;//stride of image

input clk, reset;
input [0:D*H*W*DATA_WIDTH-1] signal;//1024
input [0:D*F*DATA_WIDTH-1] filter;//64
input [0:DATA_WIDTH-1] bias;
output reg [0:(((W+2*P-F)/S)+1)*DATA_WIDTH-1] outputConv; // 128

wire [0:(((W+2*P-F)/S)+1)*DATA_WIDTH-1] outputConvUnits; // 64 窗口卷积的输出和行选择器的输入  output of the conv units and input to the row selector
//reg [0:(((W+2*P-F)/S)+1)*DATA_WIDTH-1] outputConvUnits_reg1;
//reg [0:(((W+2*P-F)/S)+1)*DATA_WIDTH-1] outputConvUnits_reg2;

wire [0:(D*H*W+P*2)*DATA_WIDTH-1] data_padding;
reg internalReset_CU;
wire [0:(((W+2*P-F)/S)+1)*F*DATA_WIDTH-1] receptiveField; // 要发送到窗口卷积的矩阵数组  array of the matrices to be sent to conv units


reg [7:0] counter;
//integer outputCounter;
//counter:完成窗口所需的时钟周期数  number of clock cycles need for the conv unit to finsish
//outputCounter: 将窗口的输出赋值到模块输出的索引  index to map the output of the conv units to the output of the module

assign data_padding = {{P{16'b0}}, signal, {P{16'b0}}};

RFselector
#(
	.DATA_WIDTH(DATA_WIDTH),
	.D(D),
	.H(H),
	.W(W),
    .P(P),
	.F(F)
) RF
(
	.signal(data_padding),
	.receptiveField(receptiveField)
);

genvar n;
generate //生成n个卷积单元
	for (n = 0; n < 128; n = n + 1) begin 
		convUnit
		#(
			.D(D),
			.F(F)
		) CU
		(
			.clk(clk),
			.reset(internalReset_CU),
			.signal(receptiveField[n*D*F*DATA_WIDTH+:D*F*DATA_WIDTH]),
			.filter(filter),
			.bias(bias),
			.result(outputConvUnits[n*DATA_WIDTH+:DATA_WIDTH])
		);
	end
endgenerate

always @ (posedge clk or negedge reset) begin
	if (reset == 1'b0) 
	begin
		internalReset_CU <= 1'b0;
		counter <= 0;
		/////////////////////////////////////////
		outputConv <= 0;
		/////////////////////////////////////////
		//outputCounter <= 0;
	end else begin
	       if(counter == 0)
	           counter <= counter + 1;
	       else if(counter == D*F+2)
	           counter <= 0;
	       
	        if (counter == D*F+2) begin//1*64+2个周期完成窗口卷积  The conv unit finishes ater 1*5*5+2 clock cycles         
			     //outputCounter <= outputCounter + 1;
			     counter <= 0;
			     /////////////////////////////////////////
			     outputConv <= outputConvUnits;
			     /////////////////////////////////////////
			     internalReset_CU <= 1'b0;  
	        end else begin
//			   /////////////////////////////////////////
//			   outputConv[0:(((W+2*P-F)/S)+1)*DATA_WIDTH] <= outputConvUnits;
//			   /////////////////////////////////////////	        
		       internalReset_CU <= 1'b1;
		       counter <= counter + 1;
	        end
	end
end

//always @ (posedge clk or negedge reset) begin
//	if (reset == 1'b0) begin
//        outputConvUnits_reg2<=0;
//        outputConv<=0;
//    end else begin
//        outputConvUnits_reg2 <= outputConvUnits_reg1;
//        outputConv <= outputConvUnits_reg2;
//    end
//end
//always @ (*) begin
//	outputConv[outputCounter*(((W+2*P-F)/S)+1)*DATA_WIDTH+:(((W+2*P-F)/S)+1)*DATA_WIDTH] = outputConvUnits;
//end

endmodule

