(*use_dsp="yes"*)
module fixedMult16#(
parameter RIGHT_SHIFT_BIT = 13
)
(
  input signed [15:0] a,
  input signed [15:0] b,
  output signed [15:0] result
);


wire [14:0] a_t,b_t;
wire sign;
reg signed [29:0] temp; // ��ʱ�洢��
reg [14:0] ret_t;
assign a_t = a[14:0];
assign b_t = b[14:0];
assign sign = a[15]^b[15];
always @(*) begin
    temp = a_t * b_t; // �˷�����
    ret_t = temp >> RIGHT_SHIFT_BIT; // �������8λ
end

assign result = {sign,ret_t};
endmodule

