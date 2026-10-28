% MIT License
% 
% Copyright (c) 2026 Shan Gai, Chenglong Lv, Chengjie Xie
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

classdef Quaternion
    %QUATERNION Represents a quaternion number (a + bi + cj + dk).
    %   This class provides basic arithmetic operations, norms, and utilities
    %   for quaternion algebra. It is particularly designed for multi-channel
    %   color image and video processing in the QTR-PAR framework.
    %
    %   Example:
    %       q1 = Quaternion(1, 2, 3, 4);
    %       q2 = Quaternion(0, 1, 1, 1);
    %       q_mul = q1 * q2;
    %       q_conj = conjugate(q1);
    %
    %   Reference:
    %       QTR-PAR: Quaternion Tensor Ring with Periodic Anchoring Regularization 
    %       for Multi-Dimensional Data Recovery.
    
    properties (SetAccess = protected)
        a (1,1) double = 0 % Real part
        b (1,1) double = 0 % i-imaginary part
        c (1,1) double = 0 % j-imaginary part
        d (1,1) double = 0 % k-imaginary part
    end
    
    methods
        function obj = Quaternion(a, b, c, d)
            %QUATERNION Constructs a Quaternion object.
            %   obj = Quaternion() creates a zero quaternion.
            %   obj = Quaternion(q) copies an existing quaternion.
            %   obj = Quaternion(s) creates a real quaternion from scalar s.
            %   obj = Quaternion(a, b, c, d) creates a + bi + cj + dk.
            
            if nargin == 0
                return;
            elseif nargin == 1
                if isa(a, 'Quaternion')
                    obj.a = a.a; obj.b = a.b; obj.c = a.c; obj.d = a.d;
                elseif isscalar(a) && isnumeric(a)
                    obj.a = a;
                else
                    error('Quaternion:InvalidInput', 'Single input must be a Quaternion or a numeric scalar.');
                end
            elseif nargin == 4
                if all(cellfun(@(x) isscalar(x) && isnumeric(x), {a, b, c, d}))
                    obj.a = a; obj.b = b; obj.c = c; obj.d = d;
                else
                    error('Quaternion:InvalidInput', 'All inputs must be numeric scalars.');
                end
            else
                error('Quaternion:InvalidInput', 'Constructor requires 0, 1, or 4 arguments.');
            end
        end
        
        function disp(obj)
            %DISP Displays the quaternion in a readable format.
            fprintf('  %.4f + %.4fi + %.4fj + %.4fk\n\n', obj.a, obj.b, obj.c, obj.d);
        end
        
        function str = char(obj)
            %CHAR Converts the quaternion to a formatted string.
            str = sprintf('%.4f + %.4fi + %.4fj + %.4fk', obj.a, obj.b, obj.c, obj.d);
        end
        
        function n = norm(obj)
            %NORM Computes the Frobenius norm (magnitude) of the quaternion.
            n = sqrt(obj.a^2 + obj.b^2 + obj.c^2 + obj.d^2);
        end
        
        function n = abs(obj)
            %ABS Computes the magnitude of the quaternion (alias for norm).
            n = norm(obj);
        end
        
        function q_conj = conjugate(obj)
            %CONJUGATE Computes the quaternion conjugate.
            q_conj = Quaternion(obj.a, -obj.b, -obj.c, -obj.d);
        end
        
        function q_inv = inv(obj)
            %INV Computes the inverse of the quaternion: q^{-1} = q^* / ||q||^2.
            n_sq = norm(obj)^2;
            if n_sq < eps
                error('Quaternion:DivideByZero', 'Quaternion inverse does not exist for the zero quaternion.');
            end
            q_inv = conjugate(obj) / n_sq;
        end
        
        function result = plus(obj, other)
            %PLUS Adds two quaternions or a quaternion and a scalar.
            if isa(other, 'Quaternion')
                result = Quaternion(obj.a + other.a, obj.b + other.b, obj.c + other.c, obj.d + other.d);
            elseif isscalar(other) && isnumeric(other)
                result = Quaternion(obj.a + other, obj.b, obj.c, obj.d);
            else
                error('Quaternion:InvalidInput', 'Operands must be Quaternion or scalar.');
            end
        end
        
        function result = minus(obj, other)
            %MINUS Subtracts a quaternion or scalar from a quaternion.
            if isa(other, 'Quaternion')
                result = Quaternion(obj.a - other.a, obj.b - other.b, obj.c - other.c, obj.d - other.d);
            elseif isscalar(other) && isnumeric(other)
                result = Quaternion(obj.a - other, obj.b, obj.c, obj.d);
            else
                error('Quaternion:InvalidInput', 'Operands must be Quaternion or scalar.');
            end
        end
        
        function result = mtimes(obj, other)
            %MTIMES Multiplies two quaternions (non-commutative) or a quaternion and a scalar.
            if isa(other, 'Quaternion')
                a1 = obj.a; b1 = obj.b; c1 = obj.c; d1 = obj.d;
                a2 = other.a; b2 = other.b; c2 = other.c; d2 = other.d;
                
                % Hamilton product
                result_a = a1*a2 - b1*b2 - c1*c2 - d1*d2;
                result_b = a1*b2 + b1*a2 + c1*d2 - d1*c2;
                result_c = a1*c2 - b1*d2 + c1*a2 + d1*b2;
                result_d = a1*d2 + b1*c2 - c1*b2 + d1*a2;
                
                result = Quaternion(result_a, result_b, result_c, result_d);
            elseif isscalar(other) && isnumeric(other)
                result = Quaternion(other * obj.a, other * obj.b, other * obj.c, other * obj.d);
            else
                error('Quaternion:InvalidInput', 'Operands must be Quaternion or scalar.');
            end
        end
        
        function result = times(obj, other)
            %TIMES Element-wise multiplication (mapped to mtimes for scalars).
            result = mtimes(obj, other);
        end
        
        function result = mrdivide(obj, other)
            %MRDIVIDE Divides a quaternion by a scalar or another quaternion.
            if isscalar(other) && isnumeric(other)
                result = Quaternion(obj.a / other, obj.b / other, obj.c / other, obj.d / other);
            elseif isa(other, 'Quaternion')
                result = obj * inv(other); % q1 / q2 = q1 * (q2)^{-1}
            else
                error('Quaternion:InvalidInput', 'Operands must be Quaternion or scalar.');
            end
        end
        
        function result = rdivide(obj, other)
            %RDIVIDE Element-wise right division (mapped to mrdivide).
            result = mrdivide(obj, other);
        end
        
        function array = toArray(obj)
            %TOARRAY Converts the quaternion to a 1-by-4 numeric array.
            array = [obj.a, obj.b, obj.c, obj.d];
        end
        
        function isPure = isPure(obj)
            %ISPURE Checks if the quaternion is purely imaginary (real part is zero).
            isPure = (obj.a == 0);
        end
        
        function result = isZero(obj, tol)
            %ISZERO Checks if the quaternion is zero within a specified tolerance.
            if nargin < 2, tol = 1e-10; end
            result = (abs(obj.a) < tol) && (abs(obj.b) < tol) && ...
                     (abs(obj.c) < tol) && (abs(obj.d) < tol);
        end
        
        function result = eq(obj, other)
            %EQ Checks if two quaternions are equal within the default tolerance (1e-10).
            if ~isa(other, 'Quaternion')
                result = false;
                return;
            end
            result = obj.isClose(other, 1e-10);
        end
        
        function result = isClose(obj, other, tol)
            %ISCLOSE Checks if two quaternions are close within a specified tolerance.
            if nargin < 3, tol = 1e-10; end
            if ~isa(other, 'Quaternion')
                result = false;
                return;
            end
            result = (abs(obj.a - other.a) < tol) && ...
                     (abs(obj.b - other.b) < tol) && ...
                     (abs(obj.c - other.c) < tol) && ...
                     (abs(obj.d - other.d) < tol);
        end
    end
    
    methods (Static)
        function q = fromRGB(r, g, b)
            %FROMRGB Creates a pure quaternion from R, G, B channels.
            q = Quaternion(0, r, g, b);
        end
        
        function q = random()
            %RANDOM Generates a random quaternion with components in [0, 1].
            q = Quaternion(rand(), rand(), rand(), rand());
        end
        
        function q = randomPure()
            %RANDOMPURE Generates a random pure quaternion with components in [0, 1].
            q = Quaternion(0, rand(), rand(), rand());
        end
    end
end