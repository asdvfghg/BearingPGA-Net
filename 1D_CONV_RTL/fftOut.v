module fftOut(clk,reset,s_axis_cartesian_tvalid,s_axis_cartesian_tlast,s_axis_cartesian_tuser,after_fft_data_RE,after_fft_data_IM,fftOut,m_axis_dout_tlast,m_axis_dout_tuser,m_axis_dout_tvalid);

parameter INT_BIT = 2;
input clk,reset,s_axis_cartesian_tvalid,s_axis_cartesian_tlast;
input [10:0] s_axis_cartesian_tuser;
input [27:0] after_fft_data_RE;
input [27:0] after_fft_data_IM;
output [15:0] fftOut;
output m_axis_dout_tlast;
output [10:0] m_axis_dout_tuser;
output m_axis_dout_tvalid;


//wire [15:0] half_re,half_im;
wire [27:0] fixed_re_2048,fixed_im_2048;

//single_to_half rehalf(.single(after_fft_data_RE),.half(half_re));
//single_to_half imhalf(.single(after_fft_data_IM),.half(half_im));

fixedMult28 re_mult(.a(after_fft_data_RE),.b(28'b0000000000000000000000000010),.result(fixed_re_2048));//³ý2048µÃµ½
fixedMult28 im_mult(.a(after_fft_data_IM),.b(28'b0000000000000000000000000010),.result(fixed_im_2048));



wire [15:0] fixed_re,fixed_im;
//wire [31 : 0] s_axis_cartesian_tdata;
wire m_axis_dout_tvalid,m_axis_dout_tlast;
wire [31 : 0] m_axis_dout_tdata;
wire [10:0] m_axis_dout_tuser;

wire [15:0] absfft;
reg [15:0] fixed_re_reg,fixed_im_reg;

assign fixed_re = {fixed_re_2048[27],fixed_re_2048[24:10]};
assign fixed_im = {fixed_im_2048[27],fixed_im_2048[24:10]};
//assign s_axis_cartesian_tdata = {fixed_im_2048[27],fixed_im_2048[24:10],fixed_re_2048[27],fixed_re_2048[24:10]};
assign absfft = m_axis_dout_tdata[15:0];
assign fftOut = {absfft[15],{INT_BIT{1'b0}},absfft[14:2]};

always@(posedge clk)begin
    if(reset == 0) begin
        fixed_re_reg <= 0;
        fixed_im_reg <= 0;
    end else begin
        fixed_re_reg <= fixed_re;
        fixed_im_reg <= fixed_im;       
    end  
end

cordic your_instance_name (
  .aclk(clk),                                        // input wire aclk
  .s_axis_cartesian_tvalid(s_axis_cartesian_tvalid),  // input wire 
  .s_axis_cartesian_tuser(s_axis_cartesian_tuser),    // input wire [10 : 0] s_axis_cartesian_tuser 
  .s_axis_cartesian_tlast(s_axis_cartesian_tlast),    // input wire s_axis_cartesian_tlast
  .s_axis_cartesian_tdata({fixed_im_reg,fixed_re_reg}),    // input wire [31 : 0] s_axis_cartesian_tdata
  .m_axis_dout_tvalid(m_axis_dout_tvalid),            // output wire m_axis_dout_tvalid
  .m_axis_dout_tuser(m_axis_dout_tuser),              // output wire [10 : 0] m_axis_dout_tuser
  .m_axis_dout_tlast(m_axis_dout_tlast),              // output wire m_axis_dout_tlast
  .m_axis_dout_tdata(m_axis_dout_tdata)              // output wire [31 : 0] m_axis_dout_tdata
);



//assign fixed_re_1 =(fixed_re_2048[27]==0)? fixed_re_2048[26:0]:(28'h8000000-fixed_re_2048); 
//assign fixed_im_1 =(fixed_im_2048[27]==0)? fixed_im_2048[26:0]:(28'h8000000-fixed_im_2048);

//assign fixed_re_2 = {fixed_re_2048[27], fixed_re_1[25:11]};
//assign fixed_im_2 = {fixed_im_2048[27], fixed_im_1[25:11]};

//fixedMult16 re_mult_sq(.a(fixed_re_2),.b(fixed_re_2),.result(fixed_re_sq));
//fixedMult16 im_mult_sq(.a(fixed_im_2),.b(fixed_im_2),.result(fixed_im_sq));

//fixedAdd16 re_add_im(.a(fixed_re_sq),.b(fixed_im_sq),.result(fftOut));
//assign fixed_re_3 = {14'b11111111111111,~(fixed_re_1[25:13])};
//assign fixed_im_3 = {14'b11111111111111,~(fixed_im_1[25:13])};


//assign fixed_re_3 = (fixed_re_2048[27]==0)?fixed_re_2[25:13]:~((fixed_re_2[25:13])-1);
//assign fixed_re_4 = (fixed_re_2048[27]==0)?{fixed_re_2048[27],14'b0,fixed_re_2048[25:13]}:{fixed_re_2048[27],fixed_re_3+1};
//assign fixed_im_4 = (fixed_im_2048[27]==0)?{fixed_im_2048[27],14'b0,fixed_im_2048[25:13]}:{fixed_im_2048[27],fixed_im_3+1};


endmodule
