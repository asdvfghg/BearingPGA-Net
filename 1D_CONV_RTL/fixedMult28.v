`timescale 1ns / 1ps
//(*use_dsp48="yes"*)

module fixedMult28(a,b,result);
input signed [27:0] a;
input signed [27:0] b;
output signed [27:0] result;

assign result = $signed(a)*$signed(b);
endmodule
