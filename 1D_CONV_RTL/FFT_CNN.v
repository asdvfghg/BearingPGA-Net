module FFT_CNN(sys_clk,reset,error_counter);//,FFT_IN,CNN_OUT

input sys_clk,reset;
//input [15:0] FFT_IN;
output reg [3:0] error_counter;
wire [3:0] CNN_OUT;

wire [15:0] FFT_IN;

wire [16383:0] cnn_in;
wire FFTData_ready,fft_last;//FFT_Flag,
reg s_axis_data_tvalid;
reg FFTreset,CNNreset,data_reset;

//reg [16383:0] cnn_in_r; 
wire locked;
integer timer_counter;
wire clk;


wire [18:0] address;

clk_10MHz CLK_10
   (
    // Clock out ports
    .clk_50(clk),     // output clk_out1
    // Status and control signals
    .resetn(1), // input reset
    .locked(locked),       // output locked
   // Clock in ports
    .clk_200(sys_clk)); 


signal_data FFTin
(
    .clk(clk),
    .reset(data_reset), 
    .signal_in(FFT_IN),
    .FFTData_ready(FFTData_ready),
    .address(address));

conv_fft FFT
(
    .clk(clk),
    .reset(FFTreset),
    .s_axis_data_tvalid(s_axis_data_tvalid),
    .signal_in(FFT_IN),
    .cnn_input(cnn_in),
    .done_flag(fft_last));
 
integrationConv CONV
(
    .clk(clk),
    .reset(CNNreset),
    .CNNinput(cnn_in),
    .CNNoutput(CNN_OUT));

always@(posedge clk) begin
    if(reset == 0) begin
        s_axis_data_tvalid <= 0;
    end else
        s_axis_data_tvalid <= FFTData_ready;
end


always@(posedge clk) begin
    if(reset == 0) begin
        CNNreset = 0;
    end else if(fft_last == 1)
        CNNreset = 1;
    else if(CNNreset == 1&&s_axis_data_tvalid == 1)
        CNNreset = 0;
    else
        CNNreset = CNNreset;
end

always@(posedge clk) begin
    if(reset == 0) begin
        FFTreset <= 0;
        data_reset <= 0;
    end else begin
        data_reset <= 1'b1;        
        FFTreset <= 1'b1;
    end
end

reg flag_1,flag_2,flag;
//reg [3:0] error_counter;

always@(posedge clk)begin
    if(reset == 0) begin
        flag_1 <= 0;
        flag_2 <= 0;
    end else if(CNN_OUT != 0) begin
        flag_1 <= 1;
        flag_2 <= flag_1;
	end else begin
	    flag_1 <= 0;
        flag_2 <= 0;
	end
end

always@(posedge clk)begin
    if(reset == 0) begin
        flag <= 0;
    end else if(flag_1 == 1 && flag_2 == 0)
        flag <= 1;
    else 
        flag <= 0;
end


always@(posedge clk)begin
    if(reset == 0)
        error_counter <= 0;
    else if(flag == 1 && CNN_OUT != 7 && address <=512000 && error_counter < 4'd15) begin
        error_counter <= error_counter+1;
	end else 
	    error_counter <= error_counter;
end


//always@(posedge clk) begin
//    if(FFT_Flag == 1'b1) begin
//        cnn_in_r <= cnn_in;
//    end else
//        cnn_in_r <= 0;
//end


//always@(posedge clk) begin
//    if(reset == 0) begin
//        FFTreset <= 0;
//        CNNreset <= 0;
//        data_reset <= 0;
//    end else begin
//        if(FFT_Flag == 1'b0) begin
//            data_reset <= 1'b1;        
//            FFTreset <= 1'b1;
//        end
        
//        if(FFT_Flag == 1'b1) begin
//            CNNreset <= 1'b1;
//            //FFTreset = 1'b0;               
//        end
//    end
//end



//always@(posedge clk) begin
//    if(reset == 0) begin
//        FFTreset <= 0;
//        data_reset <= 0;
//    end else if(FFT_Flag == 1'b0) begin
//        data_reset <= 1'b1;        
//        FFTreset <= 1'b1;
//    end
//end

//ila_0 ila(
//	.clk(sys_clk), // input wire clk
//	.probe0(CNN_OUT), // input wire [3:0]  probe0  
//	.probe1(FFT_IN), // input wire [15:0]  probe1 
//	.probe2(FFT_Flag), // input wire [0:0]  probe2 
//	.probe3(FFTreset), // input wire [0:0]  probe3 
//	.probe4(CNNreset), // input wire [0:0]  probe4 
//	.probe5(data_reset)// input wire [0:0]  probe5 
//);

//always@(posedge clk) begin
//    if(reset == 0) begin
//        FFTreset = 0;
//        CNNreset = 0;
//        data_reset = 0;
//        timer_counter = 0;
//    end else begin
//        timer_counter = timer_counter + 1;
//        if(timer_counter >0 && timer_counter <15100) begin
//            data_reset = 1'b1;                
//            FFTreset = 1'b1;
//            CNNreset = 1'b0;
//        end else if(timer_counter >15100 && timer_counter < 16000) begin
//                CNNreset = 1'b1;
////                data_reset = 1'b0;
//                FFTreset = 1'b0;               
//        end else if(timer_counter >16000)
//            timer_counter = 0;

//    end
//end
endmodule
