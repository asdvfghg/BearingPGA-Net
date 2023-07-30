module classification(clk,reset,datain,dataout);//,default_cnt

parameter DATA_WIDTH = 16;
parameter NODES = 10;

input clk,reset;
input signed [DATA_WIDTH*NODES-1:0] datain;
output reg [3:0] dataout;
//output reg [3:0] default_cnt;

wire signed [DATA_WIDTH-1:0] num_reg [0:9];
reg signed [DATA_WIDTH-1:0] max_num;


genvar i; 
  for(i = 0; i < 10; i = i + 1) 
  begin:for_assign
        assign  num_reg[9-i] = datain[i*DATA_WIDTH+:DATA_WIDTH];
end

integer j;
always@(posedge clk) begin
    if(reset == 0) begin
        max_num = 16'b0;
    end else begin
        for(j=0; j<10; j=j+1)begin
            if(num_reg[j]>max_num)
                max_num = num_reg[j];
            else
                max_num = max_num;
        end
    end
end


always@(posedge clk) begin
    if(reset == 0)
        dataout <= 4'b0000;
    else if(max_num == datain[15:0])
        dataout <= 4'b1010;
    else if(max_num == datain[31:16])
        dataout <= 4'b1001;
    else if(max_num == datain[47:32])
        dataout <= 4'b1000;
    else if(max_num == datain[63:48])
        dataout <= 4'b0111;
    else if(max_num == datain[79:64])
        dataout <= 4'b0110;
    else if(max_num == datain[95:80])
        dataout <= 4'b0101;
    else if(max_num == datain[111:96])
        dataout <= 4'b0100;
    else if(max_num == datain[127:112])
        dataout <= 4'b0011;
    else if(max_num == datain[143:128])
        dataout <= 4'b0010;
    else if(max_num == datain[159:144])
        dataout <= 4'b0001;
    else 
        dataout <= 4'b0000;
end


//always@(posedge clk) begin
//    if(reset == 0)
//        default_cnt <= 4'b0000;
//    else if(dataout != 4'd1)
//        default_cnt <= default_cnt + 1;
//end


endmodule
