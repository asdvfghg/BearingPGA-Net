`timescale 1ns / 1ps
module conv_fft(clk,reset,s_axis_data_tvalid,signal_in,cnn_input,done_flag);//

input clk,reset,s_axis_data_tvalid;
input [15:0] signal_in;
output reg [16383:0] cnn_input;
output reg done_flag;


//reg [31:0] signal_out;
wire FFT_Flag;
wire mixer_singal_tready;
wire [63:0] after_fft_data;
wire [15:0] m_axis_data_tuser;
wire m_axis_data_tvalid;
         
fft c_fft(
    .aclk(clk),// input wire aclk
    .aresetn(reset),
    .s_axis_config_tdata('d1),// input wire [23 : 0] s_axis_config_tdata
    .s_axis_config_tvalid(1),// input wire s_axis_config_tvalid
    .s_axis_config_tready(),// output wire s_axis_config_tready
    .s_axis_data_tdata({16'd0,signal_in}),//数据输入// input wire [31 : 0] s_axis_data_tdata
    .s_axis_data_tvalid(s_axis_data_tvalid),// input wire s_axis_data_tvalid
    .s_axis_data_tready(mixer_singal_tready),// output wire s_axis_data_tready
    .s_axis_data_tlast(0),// input wire s_axis_data_tlast
    .m_axis_data_tdata(after_fft_data),//数据输出 // output wire [31 : 0] m_axis_data_tdata
    .m_axis_data_tuser(m_axis_data_tuser),// output wire [15 : 0] m_axis_data_tuser
    .m_axis_data_tvalid(m_axis_data_tvalid),// output wire m_axis_data_tvalid
    .m_axis_data_tready(1),// input wire m_axis_data_tready
    .m_axis_data_tlast(FFT_Flag),// output wire m_axis_data_tlast
    .event_frame_started(),// output wire event_frame_started
    .event_tlast_unexpected(),// output wire event_tlast_unexpected
    .event_tlast_missing(),// output wire event_tlast_missing
    .event_status_channel_halt(),// output wire event_status_channel_halt
    .event_data_in_channel_halt(),// output wire event_data_in_channel_halt
    .event_data_out_channel_halt()// output wire event_data_out_channel_halt
    );
wire [27:0] after_fft_data_RE,after_fft_data_IM;//fft_data_r,fft_data_i
wire [15:0] fftOut;

assign after_fft_data_RE = after_fft_data[27:0];
assign after_fft_data_IM = after_fft_data[59:32];

reg FFT_Flag_reg1,FFT_Flag_reg2;
reg [27:0] after_fft_data_RE_reg,after_fft_data_IM_reg;
reg [15:0] fftout_reg1,fftout_reg2;
reg [15:0] m_axis_data_tuser_reg1,m_axis_data_tuser_reg2;

reg m_axis_data_tvalid_1,m_axis_data_tvalid_2;//,m_axis_data_tvalid_3,m_axis_data_tvalid_4
wire fftout_last,fftout_valid;
wire [10:0] fftout_user;

always@(posedge clk) begin
    if(reset == 1'b0)begin
        after_fft_data_RE_reg <= 0;
        after_fft_data_IM_reg <= 0;
    end else
        after_fft_data_RE_reg <= after_fft_data_RE;
        after_fft_data_IM_reg <= after_fft_data_IM;
end

fftOut FFTOUT_inst
(
.clk(clk),
.reset(reset),
.s_axis_cartesian_tuser(m_axis_data_tuser_reg2),
.s_axis_cartesian_tlast(FFT_Flag_reg2),
.s_axis_cartesian_tvalid(m_axis_data_tvalid_2),
.after_fft_data_RE(after_fft_data_RE_reg),
.after_fft_data_IM(after_fft_data_IM_reg),
.fftOut(fftOut),
.m_axis_dout_tlast(fftout_last),
.m_axis_dout_tuser(fftout_user),
.m_axis_dout_tvalid(fftout_valid));


always@(posedge clk) begin
    if(reset == 1'b0)
        fftout_reg1 <= 16'd0;
    else 
        fftout_reg1 <= fftOut;
end

always@(posedge clk) begin
    if(reset == 1'b0)
        fftout_reg2 <= 16'd0;
    else 
        fftout_reg2 <= fftout_reg1;
end


always@(posedge clk) begin
    if(reset == 1'b0) begin
        FFT_Flag_reg1 <= 0;
        FFT_Flag_reg2 <= 0;
        m_axis_data_tuser_reg1 <= 0;
        m_axis_data_tuser_reg2 <= 0;
        m_axis_data_tvalid_1 <= 0;
        m_axis_data_tvalid_2 <= 0;
//        m_axis_data_tvalid_3 <= 0;
//        m_axis_data_tvalid_4 <= 0;
    end else begin
        FFT_Flag_reg1 <= FFT_Flag;
        FFT_Flag_reg2 <= FFT_Flag_reg1;
        m_axis_data_tuser_reg1 <= m_axis_data_tuser;
        m_axis_data_tuser_reg2 <= m_axis_data_tuser_reg1;
        m_axis_data_tvalid_1 <= m_axis_data_tvalid;
        m_axis_data_tvalid_2 <= m_axis_data_tvalid_1;
//        m_axis_data_tvalid_3 <= m_axis_data_tvalid_2;
//        m_axis_data_tvalid_4 <= m_axis_data_tvalid_3;
    end
end

always@(posedge clk) begin
    if(reset == 1'b0) begin
        done_flag <= 0;
    end else 
        done_flag <= fftout_last;
end


always@(posedge clk) begin
    if(reset == 1'b0)
        cnn_input = 16384'd0;
    else if(fftout_user>=1023&&fftout_valid == 1)
        cnn_input = {cnn_input[16*1023-1:0],fftOut};
    else
        cnn_input = cnn_input; 
end


//always@(posedge clk) begin
//    if(reset == 1'b0)
//        cnn_data_ready <= 1'b0;
//    else if(m_axis_data_tuser>=1&&m_axis_data_tuser<=1024)
//        cnn_data_ready <= 1'b1;     
//    else
//        cnn_data_ready <= 1'b0;   
//end

//always@(posedge clk) begin
//    if(reset == 0) begin
//        done_flag = 0;
//    end else if(fft_last == 1)
//        done_flag = 1;
//    else
//        done_flag = done_flag;
//end

endmodule

