`timescale 100 ns / 10 ps

module convUnit(clk,reset,signal,filter,bias,result);

parameter DATA_WIDTH = 16;
parameter D = 1; //depth of the filter
parameter F = 64; //size of the filter

input clk, reset;
input [0:D*F*DATA_WIDTH-1] signal, filter;
input [0:DATA_WIDTH-1] bias;
output  [0:DATA_WIDTH-1] result;

//wire [0:DATA_WIDTH-1] bias_result;
//reg [0:DATA_WIDTH-1] result_reg1,result_reg2;
reg [DATA_WIDTH-1:0] selectedInput1, selectedInput2;
wire[0:DATA_WIDTH-1] PE_Out;
reg [6:0] i;

processingElement16#(
.RIGHT_SHIFT_BIT(12)
) PE
(
    .clk(clk),
	.reset(reset),
	.floatA(selectedInput1),
	.floatB(selectedInput2),
	.result(PE_Out)
);


fixedAdd16 FADD(PE_Out,bias,result);

//always @ (posedge clk) begin
//	if (reset == 1'b0) begin
//	    result <= 0;
//        result_reg2<=0;
//        result_reg1<=0;
//    end else begin
//        result_reg1 <= bias_result;
//        result_reg2 <= result_reg1;
//        result <= result_reg2;
//    end
//end

// 为了节省硬件，卷积是按顺序计算的 The convolution is calculated in a sequential process to save hardware
//  F*F+2个周期完成窗口卷积  The result of the element wise matrix multiplication is finished after (F*F+2) cycles (2 cycles to reset the processing element and F*F cycles to accumulate the result of the F*F multiplications) 
always @ (posedge clk) begin
	if (reset == 1'b0) begin // reset
		i <= 0;
		selectedInput1 <= 0;
		selectedInput2 <= 0;
	end else if (i > D*F-1) begin //执行完64的一维卷积后输入置0   if the convolution is finished but we still wait for other blocks to finsih, send zeros to the conv unit (in case of pipelining)
		selectedInput1 <= 0;
		selectedInput2 <= 0;
	end else begin // 发送信号部分的一个元素和滤波器的一个元素进行相乘和累加  send one element of the signal part and one element of the filter to be multiplied and accumulated
		selectedInput1 <= signal[DATA_WIDTH*i+:DATA_WIDTH];
		selectedInput2 <= filter[DATA_WIDTH*i+:DATA_WIDTH];
		i <= i + 1;
	end
end

endmodule
