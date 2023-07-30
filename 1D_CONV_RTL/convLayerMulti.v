`timescale 100 ns / 10 ps

module convLayerMulti(clk,reset,signal,outputConv,done_flag);//,filters,biases

parameter DATA_WIDTH = 16;
parameter D = 1; //Depth of image and filter
parameter H = 1; //Height of image
parameter W = 1024; //Width of image
parameter F = 64; //Size of filter
parameter K = 4; //Number of filters applied
parameter P = 28;//padding of image
parameter S = 8;//stride of image

input clk, reset;
input [0:D*H*W*DATA_WIDTH-1] signal; // 1*1024*16位宽 输入图片
//input [0:K*D*F*DATA_WIDTH-1] filters; //4*1*64*16位宽 输入卷积层
//input [0:K*DATA_WIDTH-1] biases; //输入4*16=96 偏置

output reg [0:K*(((W+2*P-F)/S)+1)*DATA_WIDTH-1] outputConv = 0; //4*128*16
output reg done_flag;

wire [0:K*D*F*DATA_WIDTH-1] filters;

wire [0:K*DATA_WIDTH-1] biases = 64'b1000111100101101100010010000111010001001100101001001110101000111;


//1000000001001000_1000001011001001_1000000000000111_0010001100101111;
//1000011101010010_0000000110010010_1000001101111001_1000010110110101;
//1000000000001111_0000000000000011_1000000000000111_1000000000001011;//6
wire Cstart;

reg [0:DATA_WIDTH-1] inputBiases;
reg [0:D*F*DATA_WIDTH-1] inputFilters; //64*4 
wire [0:(((W+2*P-F)/S)+1)*DATA_WIDTH-1] outputSingleLayers;//128*16

reg internalReset;
reg [3:0] filterSet;
reg [6:0] counter; 
reg [3:0] outputCounter;//filterset选择滤波器

Conv_weights CM(
    .clk(clk),
    .reset(reset),
    .read_flag(Cstart),
    .weights(filters));

convLayerSingle #(
	.DATA_WIDTH(DATA_WIDTH),
	.D(D),
	.H(H),
	.W(W),
	.F(F)
) UUT (
    .clk(clk),
	.reset(internalReset),
	.signal(signal),
    .filter(inputFilters[0:D*F*DATA_WIDTH-1]),
    .bias(inputBiases[0:DATA_WIDTH-1]),
    .outputConv(outputSingleLayers[0:(((W+2*P-F)/S)+1)*DATA_WIDTH-1])
 );

always @ (posedge clk) begin //or negedge reset
//初始化UUT（convLayerSingle） 
	if (Cstart == 1'b0) begin
		internalReset <= 1'b0;
		filterSet <= 0;
		counter <= 0;
		done_flag <= 0;
		outputCounter <= 0;
//开始进行卷积  当计数满1569时 outputCounter自加一 停止卷积并选择第二个卷积核进行卷积  循环6次	
	end else if (filterSet < K) begin
		if (counter == D*F+3+1) begin//1569=（28*28/14*（1*5*5+3）+1）一个卷积核计算完28*28次所需要的周期（并行度为14）		
		    //////////////////////////////
			outputConv[outputCounter*(((W+2*P-F)/S)+1)*DATA_WIDTH+:(((W+2*P-F)/S)+1)*DATA_WIDTH] <= outputSingleLayers;
			//////////////////////////////
			outputCounter <= outputCounter + 1;
			counter <= 0;
			internalReset <= 1'b0;
			filterSet <= filterSet + 1;
		end else begin			
		    /////////////////////////////////////
			//outputConv[outputCounter*(((W+2*P-F)/S)+1)*DATA_WIDTH+:(((W+2*P-F)/S)+1)*DATA_WIDTH] <= outputSingleLayers;
			inputFilters <= filters[filterSet*D*F*DATA_WIDTH+:D*F*DATA_WIDTH];
	        inputBiases <= biases[filterSet*DATA_WIDTH+:DATA_WIDTH];
	        //////////////////////////////////////////////////////////
			internalReset <= 1'b1;
			counter <= counter + 1;			
		end
	end else if(filterSet == K) begin
	   done_flag <= 1'b1;
	end
end

//选择4个卷积核中的一个进行卷积，并将输出结果存入outputConv
//always @ (*) begin
//	inputFilters = filters[filterSet*D*F*DATA_WIDTH+:D*F*DATA_WIDTH];
//	inputBiases = biases[filterSet*DATA_WIDTH+:DATA_WIDTH];
//	outputConv[outputCounter*(((W+2*P-F)/S)+1)*DATA_WIDTH+:(((W+2*P-F)/S)+1)*DATA_WIDTH] = outputSingleLayers;
//end

endmodule
