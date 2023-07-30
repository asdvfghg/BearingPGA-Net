//把图像展开后的一维数组进行数据重排，然后对应分发给n个（横向卷积并行度）CU进行窗口卷积
`timescale 100 ns / 10 ps

module RFselector(signal, receptiveField);

parameter DATA_WIDTH = 16;
parameter D = 1; //Depth of the filter
parameter H = 1; //Height of the image
parameter W = 1024; //Width of the image
parameter P = 28; //Width of the image
parameter F = 64; //Size of the filter

input [0:(D*H*W+P*2)*DATA_WIDTH-1] signal;//1024+56
//input [7:0]  column;//128
output reg [0:128*64*16-1] receptiveField;//256*64

reg [7:0] address, i;

always @ (signal) begin
	   address = 0;
		for (i = 0; i < 128; i = i + 1) begin
			receptiveField[address*F*DATA_WIDTH+:F*DATA_WIDTH] = signal[i*8*DATA_WIDTH+:F*DATA_WIDTH];
			address = address + 1;
		end
//	else begin
//		for (c = 128; c < 256; c = c + 1) begin
//			receptiveField[address*F*DATA_WIDTH+:F*DATA_WIDTH] = image[c*8*DATA_WIDTH+:F*DATA_WIDTH];
//			address = address + 1;
//		end
//	end
end

endmodule

