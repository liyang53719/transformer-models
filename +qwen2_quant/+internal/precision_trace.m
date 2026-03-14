function out = precision_trace(action, varargin)
% precision_trace   Collect precision/range trace during inference

    persistent trace
    if isempty(trace)
        trace = struct('Op', {}, 'Class', {}, 'IsDlarray', {}, 'IsInteger', {}, 'Size', {}, 'Min', {}, 'Max', {}, 'AbsMax', {});
    end

    switch lower(action)
        case 'reset'
            trace = struct('Op', {}, 'Class', {}, 'IsDlarray', {}, 'IsInteger', {}, 'Size', {}, 'Min', {}, 'Max', {}, 'AbsMax', {});
            out = trace;

        case 'log'
            if numel(varargin) < 2
                error('precision_trace:InvalidLog', 'log requires opName and value');
            end

            opName = string(varargin{1});
            value = varargin{2};

            data = value;
            isDl = isa(value, 'dlarray');
            if isDl
                data = extractdata(value);
            end

            dataClass = class(data);
            if islogical(data)
                isInt = true;
            else
                isInt = isinteger(data);
            end

            dataAbs = abs(single(data));
            entry = struct();
            entry.Op = char(opName);
            entry.Class = dataClass;
            entry.IsDlarray = isDl;
            entry.IsInteger = isInt;
            entry.Size = size(data);
            entry.Min = single(min(data(:)));
            entry.Max = single(max(data(:)));
            entry.AbsMax = single(max(dataAbs(:)));

            trace(end+1) = entry; %#ok<AGROW>
            out = entry;

        case 'get'
            out = trace;

        otherwise
            error('precision_trace:UnknownAction', 'Unknown action: %s', action);
    end
end
