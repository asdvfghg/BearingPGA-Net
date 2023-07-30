module Conv_weights(clk,reset,read_flag,weights);

parameter DATA_WIDTH = 16;
parameter F = 64;
parameter K = 4;

input clk,reset;
output reg read_flag;
output reg [DATA_WIDTH*F*K-1:0] weights;

reg [2:0] address;
wire [F*DATA_WIDTH-1:0] weight;

always@(posedge clk)begin
    if(reset == 0) begin
        address <= 0;
        read_flag <= 1'b0;
    end else if(address == 3'd5) begin
        read_flag <= 1'b1;
        address <= 0;
    end else begin
        //weights[(255-address)*DATA_WIDTH+:DATA_WIDTH] <= weight;
        address <= address + 1'b1;
    end
end

conv_weights convW(
        .clka(clk),
        .ena(reset),
        .addra(address),
        .douta(weight)
        );
      
always@(posedge clk) begin
    if(read_flag == 0)
        weights[(5-address)*DATA_WIDTH*F +: DATA_WIDTH*F] = weight;
    else
        weights = weights;
end

endmodule
