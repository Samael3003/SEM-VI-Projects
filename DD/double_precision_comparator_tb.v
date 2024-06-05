module double_precision_comparator_tb;

    reg [63:0] A_64;
    reg [63:0] B_64;
    wire equal_to;
    wire less_than;
    wire greater_than;

    real A;
    real B;

    comparator uut (
        .A_64(A_64),
        .B_64(B_64),
        .equal_to(equal_to),
        .less_than(less_than),
        .greater_than(greater_than)
    );

    // Provide input values and monitor results
    initial begin
        // Test Case 1: Equal numbers (Positives)
        A = 5.4;
        B = 5.4;
        A_64 = $realtobits(A);
        B_64 = $realtobits(B);
        $monitor("Test Case 1:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 2: Equal numbers (Negatives)
        A = -5.4;
        B = -5.4;
        A_64 = $realtobits(A);
        B_64 = $realtobits(B);
        $monitor("Test Case 2:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 3: A > B (Positives)
        A = 7.2;
        B = 6.3;
        A_64 = $realtobits(A);
        B_64 = $realtobits(B);
        $monitor("Test Case 3:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 4: A > B (Negatives)
        A = -6.3;
        B = -7.2;
        A_64 = $realtobits(A);
        B_64 = $realtobits(B);
        $monitor("Test Case 4:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;
        
        // Test Case 5: A < B (Positives)
        A = 8.1;
        B = 9.0;
        A_64 = $realtobits(A);
        B_64 = $realtobits(B);
        $monitor("Test Case 5:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 6: A < B (Negatives)
        A = -9.0;
        B = -8.1;
        A_64 = $realtobits(A);
        B_64 = $realtobits(B);
        $monitor("Test Case 6:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;
        
        // Test Case 7: Zeros
        A = 0.0;
        B = 0.0; 
        A_64 = $realtobits(A); 
        B_64 = $realtobits(B);
        $monitor("Test Case 7:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 8: Infinity
        A = 0/0;
        B = 0/0;
        A_64 = $realtobits(A); 
        B_64 = $realtobits(B);
        $monitor("Test Case 8:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;

        // Test Case 9: NaN (Not a Number)
        A = 64'h7FF8000000000000;
        B = 1;
        A_64 = A;
        B_64 = $realtobits(B);
        $monitor("Test Case 9:");
        $monitor("Number1: %h", A_64);
        $monitor("Number2: %h", B_64);
        $monitor("Equal: %b", equal_to);
        $monitor("Less Than: %b", less_than);
        $monitor("Greater Than: %b", greater_than);
        #10;
        // Finish simulation
        $finish;
    end
endmodule
