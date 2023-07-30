`timescale 1ns / 1ps
//(*use_dsp="yes"*)
module fixedAdd16(a, b, result);

input  [15:0] a, b;
output [15:0] result;

wire [14:0] a_t,b_t;
reg [14:0] ret_t;

assign a_t = a[14:0];
assign b_t = b[14:0];

assign result = (a[15]^b[15])?((a[15] == 0 && b[15] == 1)?((a_t >= b_t)?{1'b0, a_t-b_t}:{1'b1, b_t-a_t}):((a_t >= b_t)?{1'b1, a_t-b_t}:{1'b0, b_t-a_t})):{a[15],a_t+b_t};

//always@(*) begin
//    if(a[15] == 0 && b[15] == 0) begin
//        ret_t = a_t+b_t;
//        result = {1'b0, ret_t};
//    end else if(a[15] == 0 && b[15] == 1) begin
//        if(a_t >= b_t) begin
//            ret_t = a_t-b_t;
//            result = {1'b0, ret_t};
//        end else begin
//            ret_t = b_t-a_t;
//            result = {1'b1, ret_t};
//        end
//    end else if(a[15] == 1 && b[15] == 0) begin
//        if(a_t > b_t) begin
//            ret_t = a_t-b_t;
//            result = {1'b1, ret_t};
//        end else begin
//            ret_t = b_t-a_t;
//            result = {1'b0, ret_t};
//        end
//    end else if(a[15] == 1 && b[15] == 1) begin
//            ret_t = a_t+b_t;
//            result = {1'b1, ret_t};
//    end    
//end
endmodule
