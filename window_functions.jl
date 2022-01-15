module WindowFunctions

    export AbstractWindow, MeanWindow, SumWindow, GaussianWindow, get_radius, construct_kernel

    abstract type AbstractWindow{T, N} end

    struct MeanWindow{T, N} <: AbstractWindow{T, N}
        radius
        val::T
    end

    """
        SumWindow(radius)

    Constructs a sum filter (sum of values at each pixel when convolved).
    """
    struct SumWindow{T, N} <: AbstractWindow{T, N}
        radius
    end

    struct GaussianWindow{T, N} <: AbstractWindow{T, N}
        radius
        sigma::T
    end

    """
        MeanWindow{T, N}(radius [, val=1/(2*radius+1)^2])

    Constructs a mean filter (i.e. convolution with it computes the mean value of
    a neighborhood at each pixel).
    """
    function MeanWindow{T, N}(radius) where {T, N}
        val=convert(T, 1/(2*radius+1)^2)
        return MeanWindow{T, N}(radius, val)
    end

    """
        GaussianWindow{T, N}(radius [, sigma=radius/1.5])

    Constructs a gaussian blur filter.
    """
    function GaussianWindow{T, N}(radius) where {T, N}
        sigma = convert(T, radius/1.5)                  # ¯\_(ツ)_/¯
        return GaussianWindow{T, N}(radius, sigma)
    end

    get_radius(self::AbstractWindow) = self.radius

    function construct_kernel(self::MeanWindow{T, N}) where {T, N}
        return fill(self.val, ntuple(i->(2*self.radius+1), N))
    end

    function construct_kernel(self::SumWindow{T, N}) where {T, N}
        return ones(T, ntuple(i->(2*self.radius+1), N))
    end

    function construct_kernel(self::GaussianWindow{T, N}) where {T, N}
        shape = ntuple(i->(2*self.radius+1), N)
        s2 = self.sigma^2
        output = Array{T, N}(undef, shape...)
        center = CartesianIndex(ntuple(i -> self.radius + 1, N))
        for idx in CartesianIndices(output)
            output[idx] = exp(-0.5*sum(Tuple(idx - center).^2)/s2)
        end
        normalizer = sum(output)
        output ./= normalizer
        return output
    end


end    #module
