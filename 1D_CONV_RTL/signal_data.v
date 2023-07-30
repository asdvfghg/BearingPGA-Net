`timescale 1ns / 1ps

module signal_data(clk, reset, signal_in, FFTData_ready,address);

parameter CLK_CNTMAX = 15'd18000;


input clk, reset;
output reg [15:0] signal_in;
output reg FFTData_ready;
output reg [18:0] address;
 
reg [13:0] timer;
//reg [18:0] address;
wire [15:0] fft_data;

reg [14:0] clk_cnt;
reg read_valid;

always@(posedge clk) begin
    if(reset == 0)begin
        clk_cnt <= 0;
    end else if(clk_cnt == CLK_CNTMAX)
        clk_cnt <= 0;
    else
        clk_cnt <= clk_cnt + 1;
end

always@(posedge clk) begin
    if(reset == 0)
        read_valid <= 0;
    else if((clk_cnt >= 0)&&(clk_cnt <= 2047)) 
        read_valid <= 1;
    else if((clk_cnt > 2047)&&(clk_cnt <= CLK_CNTMAX)) 
        read_valid <= 0;
    else 
        read_valid <= read_valid;
end

always@(posedge clk) begin
    if(reset == 0)
        FFTData_ready <= 0;
    else if((clk_cnt >= 2)&&(clk_cnt <= 2049)) 
        FFTData_ready <= 1;
    else if((clk_cnt > 2049)&&(clk_cnt <= CLK_CNTMAX)) 
        FFTData_ready <= 0;
    else if(address == 19'd511999)
        FFTData_ready <= 0;
    else
        FFTData_ready <= FFTData_ready;
end

always@(posedge clk) begin
    if(reset == 0)
        address <= 0;
    else if(read_valid == 1 && address <= 19'd512000) 
        address <= address + 1;
    else if(read_valid == 0) 
        address <= address;
    else if(address == 19'd512001)
        address <= address;
end

always@(posedge clk)begin
    if(FFTData_ready == 0 || address == 19'd512000)
        signal_in = 0;
    else 
        signal_in = fft_data;
end


//always@(posedge clk) begin
//    if(reset == 0)begin
//        timer <= 0;
//        address <= 0;
//    end else begin
//        timer <= timer + 1;
//        if(timer >= 14'd16000) begin
//            timer <= 0;
////            address <= address+2048;
////            if(address >= 20480) begin
////                address <= 0;
////            end
//        end else if (timer <= 2047) begin
//            address <= address + 1;
//        end else if(timer >= 2047)begin
//            address <= address;
//        end //else if(timer >= 2051)begin
////            signal_in <= 0;
////        end
//        if (address >= 20480) begin
//            address <= 0;
//        end
//    end
//end



//always@(posedge clk)begin
//    if(reset == 0)
//        address = 0;
//    else if(address == 15'd20479) begin
//        address = 0;
//    end else
//        address = address+1;
//end


Siganl_data DataIn(
  .clka(clk),
  .ena(reset),
  .addra(address),
  .douta(fft_data)
);
endmodule
